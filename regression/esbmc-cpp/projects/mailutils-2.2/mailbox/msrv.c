/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

/* This is an `m-server' - a universal framework for multi-process TCP
   servers. An `m-' stands for `mail-', or `multi-' or maybe `meta-',
   I don't remember what. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <syslog.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <limits.h>
#include <mailutils/cctype.h>
#include <mailutils/server.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/cfg.h>
#include <mailutils/nls.h>
#include <mailutils/daemon.h>
#include <mailutils/acl.h>

typedef RETSIGTYPE (*mu_sig_handler_t) (int);

static mu_sig_handler_t
set_signal (int sig, mu_sig_handler_t handler)
{
#ifdef HAVE_SIGACTION
  {
    struct sigaction act, oldact;
    act.sa_handler = handler;
    sigemptyset (&act.sa_mask);
    act.sa_flags = 0;
    sigaction (sig, &act, &oldact);
    return oldact.sa_handler;
  }
#else
  return signal (sig, handler);
#endif
}

#ifndef NSIG
# define NSIG 64
#endif

union m_sockaddr
{
  struct sockaddr s_sa;
  struct sockaddr_in s_in;
  struct sockaddr_un s_un;
};

struct m_default_address
{
  union m_sockaddr s;
  int len;
};

struct _mu_m_server
{
  char *ident;                   /* Server identifier, for logging purposes.*/
  int deftype;                   /* Default server type: MU_IP_TCP/MU_IP_UDP */
  mu_server_t server;            /* The server object. */
  mu_list_t srvlist;             /* A list of configured mu_ip_server_t
				    objects. It is cleared after the objects
				    are opened and attached to the server. */
  
  mu_m_server_conn_fp conn;      /* Connection handler function. */
  mu_m_server_prefork_fp prefork;/* Pre-fork function. */
  void *data;                    /* User-supplied data for conn and prefork. */
  
  int mode;                      /* Server mode: should be removed. */
  
  int foreground;                /* Should the server remain in foregorund? */
  size_t max_children;           /* Maximum number of sub-processes to run. */
  size_t num_children;           /* Current number of running sub-processes. */
  pid_t *child_pid;
  char *pidfile;                 /* Name of a PID-file. */
  struct m_default_address defaddr;  /* Default address. */
  time_t timeout;                /* Default idle timeout. */
  mu_acl_t acl;                  /* Global access control list. */

  sigset_t sigmask;              /* A set of signals to handle by the
				    m-server.  */
  mu_sig_handler_t sigtab[NSIG]; /* Keeps old signal handlers. */
  const char *(*strexit) (int);  /* Convert integer exit code to textual
				    description. */
};

struct m_srv_config        /* Configuration data for a single TCP server. */
{
  mu_m_server_t msrv;      /* Parent m-server. */  
  mu_ip_server_t tcpsrv;  /* TCP server these data are for. */
  mu_acl_t acl;            /* Access control list for this server. */ 
  int single_process;      /* Should it run as a single process? */
  int transcript;          /* Enable session transcript. */
  time_t timeout;          /* Idle timeout for this server. */
};


static int need_cleanup = 0;
static int stop = 0; /* FIXME: Must be per-m-server */
static mu_list_t m_server_list;

#define UNUSED_PID ((pid_t)-1)

static void
alloc_children (mu_m_server_t srv)
{
  int i;
  size_t size = srv->max_children * sizeof (srv->child_pid[0]);
  
  srv->child_pid = malloc (size);
  
  if (!srv->child_pid)
    {
      mu_error ("%s", mu_strerror (ENOMEM));
      abort ();
    }
  
  for (i = 0; i < srv->max_children; i++)
    srv->child_pid[i] = UNUSED_PID;
}

static void
register_child (mu_m_server_t msrv, pid_t pid)
{
  int i;
  
  msrv->num_children++;
  for (i = 0; i < msrv->max_children; i++)
    if (msrv->child_pid[i] == UNUSED_PID)
      {
	msrv->child_pid[i] = pid;
	return;
      }
  mu_error ("%s:%d: cannot find free PID slot (internal error?)",
	    __FILE__, __LINE__);
}

static int
unregister_child (mu_m_server_t msrv, pid_t pid)
{
  int i;

  msrv->num_children--;
  for (i = 0; i < msrv->max_children; i++)
    if (msrv->child_pid[i] == pid)
      {
	msrv->child_pid[i] = UNUSED_PID;
	return 0;
      }
  return 1;
}

static void
terminate_children (mu_m_server_t msrv)
{
  if (msrv->child_pid)
    {
      int i;
      
      for (i = 0; i < msrv->max_children; i++)
	if (msrv->child_pid[i] != UNUSED_PID)
	  kill (msrv->child_pid[i], SIGTERM);
    }
}

void
mu_m_server_stop (int code)
{
  stop = code;
}

struct exit_data
{
  pid_t pid;
  int status;
};

static int
m_server_cleanup (void *item, void *data)
{
  mu_m_server_t msrv = item;
  struct exit_data *datp = data;
  
  if (unregister_child (msrv, datp->pid) == 0)
    {
      if (WIFEXITED (datp->status))
	{
	  int prio = MU_DIAG_INFO;
	  int code = WEXITSTATUS (datp->status);
	  if (code == 0)
	    prio = MU_DIAG_DEBUG;
	  if (msrv->strexit)
	    mu_diag_output (prio,
			    _("process %lu finished with code %d (%s)"),
			    (unsigned long) datp->pid,
			    code,
			    msrv->strexit (code));
	  else
	    mu_diag_output (prio,
			    _("process %lu finished with code %d"),
			    (unsigned long) datp->pid,
			    code);
	}
      else if (WIFSIGNALED (datp->status))
	mu_diag_output (MU_DIAG_ERR, "process %lu terminated on signal %d",
			(unsigned long) datp->pid,
			WTERMSIG (datp->status));
      else
	mu_diag_output (MU_DIAG_ERR,
			"process %lu terminated (cause unknown)",
			(unsigned long) datp->pid);
      return 1;
    }
  return 0;
}

static int
mu_m_server_idle (void *server_data MU_ARG_UNUSED)
{
  if (need_cleanup)
    {
      struct exit_data ex;

      need_cleanup = 0;
      while ( (ex.pid = waitpid (-1, &ex.status, WNOHANG)) > 0)
	/* Iterate over all m-servers and notify them about the fact. */
	mu_list_do (m_server_list, m_server_cleanup, &ex);
    }
  return stop;
}

static RETSIGTYPE
m_srv_signal (int signo)
{
  switch (signo)
    {
    case SIGCHLD:
      need_cleanup = 1;
      break;

    default:
      stop = 1;
      break;
    }
#ifndef HAVE_SIGACTION
  signal (signo, m_srv_sigchld);
#endif
}

void
mu_m_server_create (mu_m_server_t *psrv, const char *ident)
{
  mu_m_server_t srv = calloc (1, sizeof *srv);
  if (!srv)
    {
      mu_error ("%s", mu_strerror (ENOMEM));
      exit (1);
    }
  if (ident)
    {
      srv->ident = strdup (ident);
      if (!srv->ident)
	{
	  mu_error ("%s", mu_strerror (ENOMEM));
	  exit (1);
	}
    }
  srv->deftype = MU_IP_TCP;
  MU_ASSERT (mu_server_create (&srv->server));
  mu_server_set_idle (srv->server, mu_m_server_idle);
  sigemptyset (&srv->sigmask);
  sigaddset (&srv->sigmask, SIGCHLD);
  sigaddset (&srv->sigmask, SIGINT);
  sigaddset (&srv->sigmask, SIGTERM);
  sigaddset (&srv->sigmask, SIGQUIT);
  sigaddset (&srv->sigmask, SIGHUP);
  *psrv = srv;
  if (!m_server_list)
    mu_list_create (&m_server_list);
  mu_list_append (m_server_list, srv);
}

void
mu_m_server_set_type (mu_m_server_t srv, int type)
{
  srv->deftype = type;
}

void
mu_m_server_get_type (mu_m_server_t srv, int *type)
{
  *type = srv->deftype;
}

void
mu_m_server_set_sigset (mu_m_server_t srv, sigset_t *sigset)
{
  srv->sigmask = *sigset;
  sigaddset (&srv->sigmask, SIGCHLD);
}

void
mu_m_server_get_sigset (mu_m_server_t srv, sigset_t *sigset)
{
  *sigset = srv->sigmask;
}

void
mu_m_server_set_mode (mu_m_server_t srv, int mode)
{
  srv->mode = mode;
}

void
mu_m_server_set_conn (mu_m_server_t srv, mu_m_server_conn_fp conn)
{
  srv->conn = conn;
}

void
mu_m_server_set_prefork (mu_m_server_t srv, mu_m_server_prefork_fp fun)
{
  srv->prefork = fun;
}

void
mu_m_server_set_data (mu_m_server_t srv, void *data)
{
  srv->data = data;
}

void
mu_m_server_set_max_children (mu_m_server_t srv, size_t num)
{
  srv->max_children = num;
}

int
mu_m_server_set_pidfile (mu_m_server_t srv, const char *pidfile)
{
  free (srv->pidfile);
  srv->pidfile = strdup (pidfile);
  return srv->pidfile ? 0 : errno;
}

int
mu_m_server_set_foreground (mu_m_server_t srv, int enable)
{
  srv->foreground = enable;
  return 0;
}

void
mu_m_server_set_strexit (mu_m_server_t srv, const char *(*fun) (int))
{
  srv->strexit = fun;
}

int
mu_m_server_get_srvlist (mu_m_server_t srv, mu_list_t *plist)
{
  *plist = srv->srvlist;
  return 0;
}

const char *
mu_m_server_pidfile (mu_m_server_t srv)
{
  return srv->pidfile;
}

void
mu_m_server_set_default_address (mu_m_server_t srv, struct sockaddr *sa,
				 int salen)
{
  if (salen > sizeof srv->defaddr.s)
    {
      mu_error (_("unhandled sockaddr size"));
      abort ();
    }
  memcpy (&srv->defaddr.s.s_sa, sa, salen);
  srv->defaddr.len = salen;
}

int
mu_m_server_get_default_address (mu_m_server_t srv, struct sockaddr *sa,
				 int *salen)
{
  int len;
  
  if (!sa)
    return EINVAL;
  len = srv->defaddr.len;
  if (sa)
    {
      if (*salen < len)
	return MU_ERR_BUFSPACE;
      memcpy (sa, &srv->defaddr.s.s_sa, len);
    }
  return 0;
}


void
mu_m_server_set_default_port (mu_m_server_t srv, int num)
{
  struct sockaddr_in s_in;
  s_in.sin_family = AF_INET;
  s_in.sin_addr.s_addr = htonl (INADDR_ANY);
  s_in.sin_port = htons (num);
  mu_m_server_set_default_address (srv, (struct sockaddr*) &s_in, sizeof s_in);
}

void
mu_m_server_set_timeout (mu_m_server_t srv, time_t t)
{
  srv->timeout = t;
}

int
mu_m_server_mode (mu_m_server_t srv)
{
  return srv->mode;
}

time_t
mu_m_server_timeout (mu_m_server_t srv)
{
  return srv->timeout;
}

int
mu_m_server_foreground (mu_m_server_t srv)
{
  return srv->foreground;
}

void
m_srv_config_free (void *data)
{
  struct m_srv_config *pconf = data;
  /* FIXME */
  free (pconf);
}

static int m_srv_conn (int fd, struct sockaddr *sa, int salen,
		       void *server_data, void *call_data,
		       mu_ip_server_t srv);

static struct m_srv_config *
add_server (mu_m_server_t msrv, struct sockaddr *s, int slen, int type)
{
  mu_ip_server_t tcpsrv;
  struct m_srv_config *pconf;

  MU_ASSERT (mu_ip_server_create (&tcpsrv, s, slen, type)); /* FIXME: type */
  MU_ASSERT (mu_ip_server_set_conn (tcpsrv, m_srv_conn));
  pconf = calloc (1, sizeof (*pconf));
  if (!pconf)
    {
      mu_error ("%s", mu_strerror (ENOMEM));
      exit (1);
    }
  pconf->msrv = msrv;
  pconf->tcpsrv = tcpsrv;
  pconf->single_process = 0;
  pconf->timeout = msrv->timeout;
  MU_ASSERT (mu_ip_server_set_data (tcpsrv, pconf, m_srv_config_free));
  if (!msrv->srvlist)
    MU_ASSERT (mu_list_create (&msrv->srvlist));
  MU_ASSERT (mu_list_append (msrv->srvlist, tcpsrv));
  return pconf;
}

void
mu_m_server_configured_count (mu_m_server_t msrv, size_t count)
{
  mu_list_count (msrv->srvlist, &count);
}

void
mu_m_server_begin (mu_m_server_t msrv)
{
  int i, rc;
  size_t count = 0;

  if (!msrv->child_pid)
    alloc_children (msrv);

  mu_list_count (msrv->srvlist, &count);
  if (count == 0 && msrv->defaddr.len)
    add_server (msrv, &msrv->defaddr.s.s_sa, msrv->defaddr.len, msrv->deftype);
  
  if (!msrv->foreground)
    {
      /* Become a daemon. Take care to close inherited fds and to hold
	 first three one, in, out, err   */
      if (daemon (0, 0) < 0)
	{
	  mu_error (_("failed to become a daemon: %s"), mu_strerror (errno));
	  exit (EXIT_FAILURE);
	}
    }

  if (msrv->pidfile)
    switch (rc = mu_daemon_create_pidfile (msrv->pidfile))
      {
      case 0:
	break;
	
      case EINVAL:
	mu_error (_("%s: invalid name for a pidfile"), msrv->pidfile);
	break;
	
      default:
	mu_error (_("cannot create pidfile `%s': %s"), msrv->pidfile,
		  mu_strerror (rc));
      }

  for (i = 0; i < NSIG; i++)
    if (sigismember (&msrv->sigmask, i))
      msrv->sigtab[i] = set_signal (i, m_srv_signal);
}

void
mu_m_server_restore_signals (mu_m_server_t msrv)
{
  int i;
  
  for (i = 0; i < NSIG; i++)
    if (sigismember (&msrv->sigmask, i))
      set_signal (i, msrv->sigtab[i]);
}

void
mu_m_server_end (mu_m_server_t msrv)
{
  mu_m_server_restore_signals (msrv);
}

void
mu_m_server_destroy (mu_m_server_t *pmsrv)
{
  mu_m_server_t msrv = *pmsrv;
  mu_list_remove (m_server_list, msrv);
  mu_server_destroy (&msrv->server);
  free (msrv->child_pid);
  /* FIXME: Send processes the TERM signal here?*/
  free (msrv->ident);
  free (msrv);
  *pmsrv = NULL;
}

static int
tcp_conn_handler (int fd, void *conn_data, void *server_data)
{
  mu_ip_server_t tcpsrv = (mu_ip_server_t) conn_data;
  int rc = mu_ip_server_accept (tcpsrv, server_data);
  if (rc && rc != EINTR)
    {
      mu_ip_server_shutdown (tcpsrv);
      return MU_SERVER_CLOSE_CONN;
    }
  return stop ? MU_SERVER_SHUTDOWN : MU_SERVER_SUCCESS;
}

static void
tcp_conn_free (void *conn_data, void *server_data)
{
  mu_ip_server_t tcpsrv = (mu_ip_server_t) conn_data;
  mu_ip_server_destroy (&tcpsrv);
}

static int
_open_conn (void *item, void *data)
{
  union
  {
    struct sockaddr sa;
    char pad[512];
  }
  addr;
  int addrlen = sizeof addr;
  char *p;
  mu_ip_server_t tcpsrv = item;
  mu_m_server_t msrv = data;
  int rc = mu_ip_server_open (tcpsrv);
  if (rc)
    {
      mu_ip_server_get_sockaddr (tcpsrv, &addr.sa, &addrlen);
      p = mu_sockaddr_to_astr (&addr.sa, addrlen);
      mu_error (_("cannot open connection on %s: %s"), p, mu_strerror (rc));
      free (p);
      return 0;
    }
  rc = mu_server_add_connection (msrv->server,
				 mu_ip_server_get_fd (tcpsrv),
				 tcpsrv,
				 tcp_conn_handler, tcp_conn_free);
  if (rc)
    {
      mu_ip_server_get_sockaddr (tcpsrv, &addr.sa, &addrlen);
      p = mu_sockaddr_to_astr (&addr.sa, addrlen);
      mu_error (_("cannot add connection %s: %s"), p, mu_strerror (rc));
      free (p);
      mu_ip_server_shutdown (tcpsrv);
      mu_ip_server_destroy (&tcpsrv);
    }
  return 0;
}  

int
mu_m_server_run (mu_m_server_t msrv)
{
  int rc;
  size_t count;
  mode_t saved_umask = umask (0117);
  mu_list_do (msrv->srvlist, _open_conn, msrv);
  umask (saved_umask);
  mu_list_destroy (&msrv->srvlist);
  MU_ASSERT (mu_server_count (msrv->server, &count));
  if (count == 0)
    {
      mu_error (_("no servers configured: exiting"));
      exit (1);
    }
  if (msrv->ident)
    mu_diag_output (MU_DIAG_INFO, _("%s started"), msrv->ident);
  rc = mu_server_run (msrv->server);
  terminate_children (msrv);
  if (msrv->ident)
    mu_diag_output (MU_DIAG_INFO, _("%s terminated"), msrv->ident);
  return rc;
}



int
mu_m_server_check_acl (mu_m_server_t msrv, struct sockaddr *s, int salen)
{
  if (msrv->acl)
    {
      mu_acl_result_t res;
      int rc = mu_acl_check_sockaddr (msrv->acl, s, salen, &res);
      if (rc)
	{
	  char *p = mu_sockaddr_to_astr (s, salen);
	  mu_error (_("access from %s blocked: cannot check ACLs: %s"),
		    p, mu_strerror (rc));
	  free (p);
	  return 1;
	}
      switch (res)
	{
	case mu_acl_result_undefined:
	  {
	    char *p = mu_sockaddr_to_astr (s, salen);
	    mu_diag_output (MU_DIAG_INFO,
			    _("%s: undefined ACL result; access allowed"),
			    p);
	    free (p);
	  }
	  break;
	      
	case mu_acl_result_accept:
	  break;
	      
	case mu_acl_result_deny:
	  {
	    char *p = mu_sockaddr_to_astr (s, salen);
	    mu_error (_("access from %s blocked"), p);
	    free (p);
	    return 1;
	  }
	}
    }
  return 0;
}

int
m_srv_conn (int fd, struct sockaddr *sa, int salen,
	    void *server_data, void *call_data,
	    mu_ip_server_t srv)
{
  int status;
  struct m_srv_config *pconf = server_data;
  
  if (mu_m_server_check_acl (pconf->msrv, sa, salen))
    return 0;

  if (!pconf->single_process)
    {
      pid_t pid;

      if (mu_m_server_idle (server_data))
	return MU_SERVER_SHUTDOWN;
      if (pconf->msrv->max_children
	  && pconf->msrv->num_children >= pconf->msrv->max_children)
        {
	  mu_diag_output (MU_DIAG_ERROR, _("too many children (%lu)"),
			  (unsigned long) pconf->msrv->num_children);
          pause ();
          return 0;
        }
      if (pconf->msrv->prefork
	  && pconf->msrv->prefork (fd, pconf->msrv->data, sa, salen))
	return 0;
      
      pid = fork ();
      if (pid == -1)
	mu_diag_output (MU_DIAG_ERROR, "fork: %s", strerror (errno));
      else if (pid == 0) /* Child.  */
	{
	  mu_ip_server_shutdown (srv); /* FIXME: does it harm for MU_IP_UDP? */
	  mu_m_server_restore_signals (pconf->msrv);
	  status = pconf->msrv->conn (fd, sa, salen, pconf->msrv->data, srv,
				      pconf->timeout, pconf->transcript);
	  closelog ();
	  exit (status);
	}
      else
	{
	  register_child (pconf->msrv, pid);
	}
    }
  else if (!pconf->msrv->prefork
	   || pconf->msrv->prefork (fd, pconf->msrv->data, sa, salen) == 0)
    pconf->msrv->conn (fd, sa, salen, pconf->msrv->data, srv,
		       pconf->timeout, pconf->transcript);
  return 0;
}



unsigned short
get_port (mu_debug_t debug, const char *p)
{
  if (p)
    {
      char *q;
      unsigned long n = strtoul (p, &q, 0);
      if (*q == 0)
	{
	  if (n > USHRT_MAX)
	    {
	      mu_debug_printf (debug, MU_DIAG_ERROR,
				   _("invalid port number: %s\n"), p);
	      return 1;
	    }
	  
	  return htons (n);
	}
      else
	{
	  struct servent *sp = getservbyname (p, "tcp");
	  if (!sp)
	    return 0;
	  return sp->s_port;
	}
    }
  return 0;
}

static int
get_family (const char **pstr, sa_family_t *pfamily)
{
  static struct family_tab
  {
    int len;
    char *pfx;
    int family;
  } ftab[] = {
#define S(s,f) { sizeof (#s":") - 1, #s":", f }
    S (file, AF_UNIX),
    S (unix, AF_UNIX),
    S (local, AF_UNIX),
    S (socket, AF_UNIX),
    S (inet, AF_INET),
    S (tcp, AF_INET),
#undef S
    { 0 }
  };
  struct family_tab *fp;
  
  const char *str = *pstr;
  int len = strlen (str);
  for (fp = ftab; fp->len; fp++)
    {
      if (len > fp->len && memcmp (str, fp->pfx, fp->len) == 0)
	{
	  str += fp->len;
	  if (str[0] == '/' && str[1] == '/')
	    str += 2;
	  *pstr = str;
	  *pfamily = fp->family;
	  return 0;
	}
    }
  return 1;
}

static int
is_ip_addr (const char *arg)
{
  int     dot_count;
  int     digit_count;

  dot_count = 0;
  digit_count = 0;
  for (; *arg != 0 && *arg != ':'; arg++)
    {
      if (*arg == '.')
	{
	  if (++dot_count > 3)
	    break;
	  digit_count = 0;
	}
      else if (!(mu_isdigit (*arg) && ++digit_count <= 3))
	return 0;
    }
  return dot_count == 3;
}  

int
_mu_m_server_parse_url (mu_debug_t debug, const char *arg, union m_sockaddr *s,
			int *psalen, struct sockaddr *defsa)
{
  char *p;
  unsigned short n;
  int len;
      
  if (is_ip_addr (arg))
    s->s_sa.sa_family = AF_INET;
  else if (get_family (&arg, &s->s_sa.sa_family))
    {
      mu_debug_printf (debug, MU_DIAG_ERROR, _("invalid family\n"));
      return EINVAL;
    }
      
  switch (s->s_sa.sa_family)
    {
    case AF_INET:
      *psalen = sizeof (s->s_in);
      if ((n = get_port (debug, arg)))
	{
	  s->s_in.sin_addr.s_addr = htonl (INADDR_ANY);
	  s->s_in.sin_port = htons (n);	  
	}
      else
	{
	  p = strchr (arg, ':');
	  if (p)
	    *p++ = 0;
	  if (inet_aton (arg, &s->s_in.sin_addr) == 0)
	    {
	      struct hostent *hp = gethostbyname (arg);
	      if (hp)
		s->s_in.sin_addr.s_addr = *(unsigned long *)hp->h_addr;
	      else
		{
		  mu_debug_printf (debug, MU_DIAG_ERROR,
				   _("invalid IP address: %s\n"), arg);
		  return EINVAL;
		}
	    }
	  if (p)
	    {
	      n = get_port (debug, p);
	      if (!n)
		{
		  mu_debug_printf (debug, MU_DIAG_ERROR,
				   _("invalid port number: %s\n"), p);
		  return EINVAL;
		}
	      s->s_in.sin_port = n;
	    }
	  else if (defsa && defsa->sa_family == AF_INET)
	    s->s_in.sin_port = ((struct sockaddr_in*)defsa)->sin_port;
	  else
	    {
	      mu_debug_printf (debug, MU_DIAG_ERROR,
			       _("missing port number\n"));
	      return EINVAL;
	    }
	}
      break;

    case AF_UNIX:
      *psalen = sizeof (s->s_un);
      len = strlen (arg);
      if (len > sizeof s->s_un.sun_path - 1)
	{
	  mu_error (_("%s: file name too long"), arg);
	  return EINVAL;
	}
      strcpy (s->s_un.sun_path, arg);
      break;
    }
  return 0;
}

int
mu_m_server_parse_url (mu_m_server_t msrv, char *arg,
		       struct sockaddr *sa, int *psalen)
{
  int rc;
  union m_sockaddr s;
  int salen;
  mu_debug_t debug;
  
  mu_diag_get_debug (&debug);
  rc = _mu_m_server_parse_url (debug, arg, &s, &salen, &msrv->defaddr.s.s_sa);
  if (rc)
    return rc;
  if (sa)
    {
      if (*psalen < salen)
	return MU_ERR_BUFSPACE;
      memcpy (sa, &s.s_sa, salen);
    }
  *psalen = salen;
  return 0;
}

static int
server_block_begin (mu_debug_t debug, const char *arg, mu_m_server_t msrv,
		    void **pdata)
{
  union m_sockaddr s;
  int salen;
  if (_mu_m_server_parse_url (debug, arg, &s, &salen, &msrv->defaddr.s.s_sa))
    return 1;
  *pdata = add_server (msrv, &s.s_sa, salen, msrv->deftype);
  return 0;
}

static int
server_section_parser (enum mu_cfg_section_stage stage,
		       const mu_cfg_node_t *node,
		       const char *section_label, void **section_data,
		       void *call_data,
		       mu_cfg_tree_t *tree)
{
  switch (stage)
    {
    case mu_cfg_section_start:
      {
	if (node->label == NULL || node->label->type != MU_CFG_STRING)
	  return 1;
	/* FIXME: should not modify 2nd arg, or it should not be const */
	return server_block_begin (tree->debug, node->label->v.string,
				   *section_data, section_data);
      }
      break;

    case mu_cfg_section_end:
      {
	struct m_srv_config *pconf = *section_data;
	if (pconf->acl)
	  mu_ip_server_set_acl (pconf->tcpsrv, pconf->acl);
      }
      break;
    }
  return 0;
}

static int
_cb_daemon_mode (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  int *pmode = data;
  
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (strcmp (val->v.string, "inetd") == 0
      || strcmp (val->v.string, "interactive") == 0)
    *pmode = MODE_INTERACTIVE;
  else if (strcmp (val->v.string, "daemon") == 0)
    *pmode = MODE_DAEMON;
  else
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("unknown daemon mode"));
      return 1;
    }
  return 0;
}

static int
_cb_port (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  struct m_default_address *ap = data;
  unsigned short num;

  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  num = get_port (debug, val->v.string);
  if (!num)
    return 1;
  ap->s.s_in.sin_family = AF_INET;
  ap->s.s_in.sin_addr.s_addr = htonl (INADDR_ANY);
  ap->s.s_in.sin_port = num;
  ap->len = sizeof ap->s.s_in;
  return 0;
}

static struct mu_cfg_param dot_server_cfg_param[] = {
  { "max-children", mu_cfg_size,
    NULL, mu_offsetof (struct _mu_m_server,max_children), NULL,
    N_("Maximum number of children processes to run simultaneously.") },
  { "mode", mu_cfg_callback,
    NULL, mu_offsetof (struct _mu_m_server,mode), _cb_daemon_mode,
    N_("Set daemon mode (either inetd (or interactive) or daemon)."),
    N_("mode") },
  { "foreground", mu_cfg_bool,
    NULL, mu_offsetof (struct _mu_m_server, foreground), NULL,
    N_("Run in foreground.") },
  { "pidfile", mu_cfg_string,
    NULL, mu_offsetof (struct _mu_m_server,pidfile), NULL,
    N_("Store PID of the master process in this file."),
    N_("file") },
  { "port", mu_cfg_callback,
    NULL, mu_offsetof (struct _mu_m_server,defaddr), _cb_port,
    N_("Default port number.") },
  { "timeout", mu_cfg_time,
    NULL, mu_offsetof (struct _mu_m_server,timeout), NULL,
    N_("Set idle timeout.") },
  { "server", mu_cfg_section, NULL, 0, NULL,
    N_("Server configuration.") },
  { "acl", mu_cfg_section, NULL, mu_offsetof (struct _mu_m_server,acl), NULL,
    N_("Per-server access control list") },
  { NULL }
};
    
static struct mu_cfg_param server_cfg_param[] = {
  { "single-process", mu_cfg_bool, 
    NULL, mu_offsetof (struct m_srv_config, single_process), NULL,
    N_("Run this server in foreground.") },
  { "transcript", mu_cfg_bool,
    NULL, mu_offsetof (struct m_srv_config, transcript), NULL,
    N_("Log the session transcript.") },
  { "timeout", mu_cfg_time,
    NULL, mu_offsetof (struct m_srv_config, timeout), NULL,
    N_("Set idle timeout.") },
  { "acl", mu_cfg_section,
    NULL, mu_offsetof (struct m_srv_config, acl), NULL,
    N_("Global access control list.") },
  { NULL }
};

void
mu_m_server_cfg_init ()
{
  struct mu_cfg_section *section;
  if (mu_create_canned_section ("server", &section) == 0)
    {
      section->parser = server_section_parser;
      section->label = N_("ipaddr[:port]");
      mu_cfg_section_add_params (section, server_cfg_param);
    }
  if (mu_create_canned_section (".server", &section) == 0)
    {
      mu_cfg_section_add_params (section, dot_server_cfg_param);
    }
}



