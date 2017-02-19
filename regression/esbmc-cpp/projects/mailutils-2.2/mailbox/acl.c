/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <arpa/inet.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/acl.h>
#include <mailutils/argcv.h>
#include <mailutils/list.h>
#include <mailutils/debug.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/kwd.h>
#include <mailutils/vartab.h>
#include <mailutils/io.h>

struct _mu_acl_entry
{
  mu_acl_action_t action;
  void *arg;
  unsigned netmask;
  int salen;
  struct sockaddr sa[1];
};

struct _mu_acl
{
  mu_debug_t debug;
  mu_list_t aclist;
};


static void
_destroy_acl_entry (void *item)
{
  struct _mu_acl_entry *p = item;
  free (p);
  /* FIXME: free arg? */
}

static size_t
mu_acl_entry_size (int salen)
{
  return sizeof (struct _mu_acl_entry) + salen - sizeof (struct sockaddr);
}

static int
prepare_sa (struct sockaddr *sa)
{
  switch (sa->sa_family)
    {
    case AF_INET:
      {
	struct sockaddr_in *s_in = (struct sockaddr_in *)sa;
	s_in->sin_addr.s_addr = ntohl (s_in->sin_addr.s_addr);
	break;
      }
      
    case AF_UNIX:
      break;

    default:
      return 1;
    }
  return 0;
}

int
mu_acl_entry_create (struct _mu_acl_entry **pent,
		     mu_acl_action_t action, void *data,
		     struct sockaddr *sa, int salen, unsigned long netmask)
{
  struct _mu_acl_entry *p = malloc (mu_acl_entry_size (salen));
  if (!p)
    return EINVAL;

  p->action = action;
  p->arg = data;
  p->netmask = ntohl (netmask);
  p->salen = salen;
  memcpy (p->sa, sa, salen);
  if (prepare_sa (p->sa))
    {
      free (p);
      return EINVAL;
    }
  *pent = p;
  return 0;
}


int
mu_acl_create (mu_acl_t *pacl)
{
  int rc;
  mu_acl_t acl;
  mu_log_level_t level;

  acl = calloc (1, sizeof (*acl));
  if (!acl)
    return errno;
  rc = mu_list_create (&acl->aclist);
  if (rc)
    free (acl);
  else
    *pacl = acl;
  mu_list_set_destroy_item (acl->aclist, _destroy_acl_entry);

  level = mu_global_debug_level ("acl");
  if (level)
    {
      int status = mu_debug_create (&acl->debug, NULL);
      if (status == 0)
	mu_debug_set_level (acl->debug, level);
    }
  
  return rc;
}

int
mu_acl_count (mu_acl_t acl, size_t *pcount)
{
  if (!acl)
    return EINVAL;
  return mu_list_count (acl->aclist, pcount);
}

int
mu_acl_destroy (mu_acl_t *pacl)
{
  mu_acl_t acl;
  if (!pacl || !*pacl)
    return EINVAL;
  acl = *pacl;
  mu_list_destroy (&acl->aclist);
  mu_debug_destroy (&acl->debug, NULL);
  free (acl);
  *pacl = acl;
  return 0;
}
		   
int
mu_acl_get_debug (mu_acl_t acl, mu_debug_t *pdebug)
{
  if (!acl)
    return EINVAL;
  if (!pdebug)
    return MU_ERR_OUT_NULL;
  *pdebug = acl->debug;
  return 0;
}

int
mu_acl_set_debug (mu_acl_t acl, mu_debug_t debug)
{
  if (!acl)
    return EINVAL;
  acl->debug = debug;
  return 0;
}

int
mu_acl_get_iterator (mu_acl_t acl, mu_iterator_t *pitr)
{
  if (!acl)
    return EINVAL;
  return mu_list_get_iterator (acl->aclist, pitr);
}

int
mu_acl_append (mu_acl_t acl, mu_acl_action_t act,
	       void *data, struct sockaddr *sa, int salen,
	       unsigned long netmask)
{
  int rc;
  struct _mu_acl_entry *ent;
  
  if (!acl)
    return EINVAL;
  rc = mu_acl_entry_create (&ent, act, data, sa, salen, netmask);
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot allocate ACL entry: %s",
		 mu_strerror (rc));
      return ENOMEM;
    }
  
  rc = mu_list_append (acl->aclist, ent);
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot append ACL entry: %s",
		 mu_strerror (rc));
      free (ent);
    }
  return rc;
}

int
mu_acl_prepend (mu_acl_t acl, mu_acl_action_t act, void *data,
		struct sockaddr *sa, int salen, unsigned long netmask)
{
  int rc;
  struct _mu_acl_entry *ent;
  
  if (!acl)
    return EINVAL;
  rc = mu_acl_entry_create (&ent, act, data, sa, salen, netmask);
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot allocate ACL entry: %s",
		 mu_strerror (rc));
      return ENOMEM;
    }
  rc = mu_list_prepend (acl->aclist, ent); 
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot prepend ACL entry: %s",
		 mu_strerror (rc));
      free (ent);
    }
  return rc;
}

int
mu_acl_insert (mu_acl_t acl, size_t pos, int before,
	       mu_acl_action_t act, void *data,
	       struct sockaddr *sa, int salen, unsigned long netmask)
{
  int rc;
  void *ptr;
  struct _mu_acl_entry *ent;
  
  if (!acl)
    return EINVAL;
  
  rc = mu_list_get (acl->aclist, pos, &ptr);
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "No such entry %lu",
		 (unsigned long) pos);
      return rc;
    }
  rc = mu_acl_entry_create (&ent, act, data, sa, salen, netmask);
  if (!ent)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot allocate ACL entry: %s",
		 mu_strerror (rc));
      return ENOMEM;
    }
  rc = mu_list_insert (acl->aclist, ptr, ent, before);
  if (rc)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, "Cannot insert ACL entry: %s",
		 mu_strerror (rc));
      free (ent);
    }
  return rc;
}


static mu_kwd_t action_tab[] = {
  { "accept", mu_acl_accept },
  { "deny", mu_acl_deny },
  { "log", mu_acl_log },
  { "exec", mu_acl_exec },
  { "ifexec", mu_acl_ifexec },
  { NULL }
};

int
mu_acl_action_to_string (mu_acl_action_t act, const char **pstr)
{
  return mu_kwd_xlat_tok (action_tab, act, pstr);
}

int
mu_acl_string_to_action (const char *str, mu_acl_action_t *pres)
{
  int x;
  int rc = mu_kwd_xlat_name (action_tab, str, &x);
  if (rc == 0)
    *pres = x;
  return rc;
}

#define MU_S_UN_NAME(sa, salen) \
  ((salen < mu_offsetof (struct sockaddr_un,sun_path)) ? "" : (sa)->sun_path)

static void
debug_sockaddr (mu_debug_t dbg, mu_log_level_t lvl, struct sockaddr *sa,
		int salen)
{
  switch (sa->sa_family)
    {
    case AF_INET:
      {
	struct sockaddr_in s_in = *(struct sockaddr_in *)sa;
	s_in.sin_addr.s_addr = htonl (s_in.sin_addr.s_addr);
	mu_debug_printf (dbg, lvl, "{AF_INET %s:%d}",
			 inet_ntoa (s_in.sin_addr), ntohs (s_in.sin_port));
	break;
      }

    case AF_UNIX:
      {
	struct sockaddr_un *s_un = (struct sockaddr_un *)sa;
	if (MU_S_UN_NAME(s_un, salen)[0] == 0)
	  mu_debug_printf (dbg, lvl, "{AF_UNIX}");
	else
	  mu_debug_printf (dbg, lvl, "{AF_UNIX %s}", s_un->sun_path);
	break;
      }

    default:
      mu_debug_printf (dbg, lvl, "{Unsupported family: %d}", sa->sa_family);
    }
}

size_t
mu_stpcpy (char **pbuf, size_t *psize, const char *src)
{
  size_t slen = strlen (src);
  if (pbuf == NULL || *pbuf == NULL)
    return slen;
  else
    {
      char *buf = *pbuf;
      size_t size = *psize;
      if (size > slen)
	size = slen;
      memcpy (buf, src, size);
      *psize -= size;
      *pbuf += size;
      if (*psize)
	**pbuf = 0;
      else
	(*pbuf)[-1] = 0;
      return size;
    }
}

void
mu_sockaddr_to_str (const struct sockaddr *sa, int salen,
		    char *bufptr, size_t buflen,
		    size_t *plen)
{
  char *nbuf;
  size_t len = 0;
  switch (sa->sa_family)
    {
    case AF_INET:
      {
	struct sockaddr_in s_in = *(struct sockaddr_in *)sa;
	len += mu_stpcpy (&bufptr, &buflen, inet_ntoa (s_in.sin_addr));
	len += mu_stpcpy (&bufptr, &buflen, ":");
	if (mu_asprintf (&nbuf, "%hu", ntohs (s_in.sin_port)) == 0)
	  {
	    len += mu_stpcpy (&bufptr, &buflen, nbuf);
	    free (nbuf);
	  }
	break;
      }

    case AF_UNIX:
      {
	struct sockaddr_un *s_un = (struct sockaddr_un *)sa;
	if (MU_S_UN_NAME(s_un, salen)[0] == 0)
	  len += mu_stpcpy (&bufptr, &buflen, "anonymous socket");
	else
	  {
	    len += mu_stpcpy (&bufptr, &buflen, "socket ");
	    len += mu_stpcpy (&bufptr, &buflen, s_un->sun_path);
	  }
	break;
      }

    default:
      len += mu_stpcpy (&bufptr, &buflen, "{Unsupported family");
      if (mu_asprintf (&nbuf, ": %d", sa->sa_family) == 0)
	{
	  len += mu_stpcpy (&bufptr, &buflen, nbuf);
	  free (nbuf);
	}
      len += mu_stpcpy (&bufptr, &buflen, "}");
    }
  if (plen)
    *plen = len + 1;
}

char *
mu_sockaddr_to_astr (const struct sockaddr *sa, int salen)
{
  size_t size;
  char *p;
  
  mu_sockaddr_to_str (sa, salen, NULL, 0, &size);
  p = malloc (size);
  if (p)
    mu_sockaddr_to_str (sa, salen, p, size, NULL);
  return p;
}

int
_acl_match (mu_debug_t debug, struct _mu_acl_entry *ent, struct sockaddr *sa,
	    int salen)
{
#define RESMATCH(word)                                   \
  if (mu_debug_check_level (debug, MU_DEBUG_TRACE0))     \
    mu_debug_printf (debug, MU_DEBUG_TRACE0, "%s; ", word);
							      
  if (mu_debug_check_level (debug, MU_DEBUG_TRACE0))
    {
      struct in_addr a;
      
      __MU_DEBUG1 (debug, MU_DEBUG_TRACE0, "%s", "Does ");
      debug_sockaddr (debug, MU_DEBUG_TRACE0, sa, salen);
      mu_debug_printf (debug, MU_DEBUG_TRACE0, " match ");
      debug_sockaddr (debug, MU_DEBUG_TRACE0, ent->sa, salen);
      a.s_addr = ent->netmask;
      a.s_addr = htonl (a.s_addr);
      mu_debug_printf (debug, MU_DEBUG_TRACE0, " netmask %s? ", inet_ntoa (a));
    }

  if (ent->sa->sa_family != sa->sa_family)
    {
      RESMATCH ("no");
      return 1;
    }

  switch (ent->sa->sa_family)
    {
    case AF_INET:
      {
	struct sockaddr_in *sin_ent = (struct sockaddr_in *)ent->sa;
	struct sockaddr_in *sin_item = (struct sockaddr_in *)sa;
	
	if (sin_ent->sin_addr.s_addr !=
	    (sin_item->sin_addr.s_addr & ent->netmask))
	  {
	    RESMATCH ("no (address differs)");
	    return 1;
	  }

	if (sin_ent->sin_port && sin_item->sin_port
	    && sin_ent->sin_port != sin_item->sin_port)
	  {
	    RESMATCH ("no (port differs)");
	    return 1;
	  }
	break;
      }
	  
    case AF_UNIX:
      {
	struct sockaddr_un *sun_ent = (struct sockaddr_un *)ent->sa;
	struct sockaddr_un *sun_item = (struct sockaddr_un *)sa;

	if (MU_S_UN_NAME (sun_ent, ent->salen)[0]
	    && MU_S_UN_NAME (sun_item, salen)[0]
	    && strcmp (sun_ent->sun_path, sun_item->sun_path))
	  {
	    RESMATCH ("no");
	    return 1;
	  }
	break;
      }
    }
  
  RESMATCH ("yes");
  return 0;
}

struct run_closure
{
  unsigned idx;
  mu_debug_t debug;
  struct sockaddr *sa;
  int salen;
  mu_acl_result_t *result;
};

static int
_expand_aclno (const char *name, void *data, char **p)
{
  struct run_closure *rp = data;
  /*FIXME: memory leak*/
  return mu_asprintf (p, "%u", rp->idx);
}

#if defined (HAVE_SYSCONF) && defined (_SC_OPEN_MAX)
# define getmaxfd() sysconf (_SC_OPEN_MAX)
#elif defined (HAVE_GETDTABLESIZE)
# define getmaxfd() getdtablesize ()
#else
# define getmaxfd() 64
#endif

static int
expand_arg (const char *cmdline, struct run_closure *rp, char **s)
{
  int rc;
  mu_vartab_t vtab;
  
  MU_DEBUG1 (rp->debug, MU_DEBUG_TRACE0, "Expanding \"%s\" => ", cmdline);
  
  mu_vartab_create (&vtab);
  mu_vartab_define_exp (vtab, "aclno", _expand_aclno, NULL, rp);
  switch (rp->sa->sa_family)
    {
    case AF_INET:
      {
	struct sockaddr_in *s_in = (struct sockaddr_in *)rp->sa;
	struct in_addr addr = s_in->sin_addr;
	char *p;
	
	mu_vartab_define (vtab, "family", "AF_INET", 1);
	addr.s_addr = htonl (addr.s_addr);
	mu_vartab_define (vtab, "address", inet_ntoa (addr), 0);
	if (mu_asprintf (&p, "%hu", ntohs (s_in->sin_port)) == 0)
	  {
	    mu_vartab_define (vtab, "port", p, 0);
	    free (p);
	  }
      }
      break;
      
    case AF_UNIX:
      {
	struct sockaddr_un *s_un = (struct sockaddr_un *)rp->sa;
	
	mu_vartab_define (vtab, "family", "AF_UNIX", 1);
	mu_vartab_define (vtab, "address", s_un->sun_path, 1);
      }
      break;
    }
  
  rc = mu_vartab_expand (vtab, cmdline, s);
  mu_vartab_destroy (&vtab);

  if (rc == 0)
    MU_DEBUG1 (rp->debug, MU_DEBUG_TRACE0, "\"%s\". ", *s);
  else
    MU_DEBUG (rp->debug, MU_DEBUG_TRACE0, "failed. ");
  return rc;
}

static int
spawn_prog (const char *cmdline, int *pstatus, struct run_closure *rp)
{
  char *s;
  pid_t pid;

  if (expand_arg (cmdline, rp, &s))
    s = strdup (cmdline);

  pid = fork ();
  if (pid == 0)
    {
      int i;
      int argc;
      char **argv;

      mu_argcv_get (s, " \t", NULL, &argc, &argv);
      for (i = getmaxfd (); i > 2; i--)
	close (i);
      execvp (argv[0], argv);
      exit (127);
    }

  free (s);

  if (pid == (pid_t)-1)
    {
      MU_DEBUG1 (rp->debug, MU_DEBUG_ERROR, "cannot fork: %s",
		 mu_strerror (errno));
      return errno;
    }

  if (pstatus)
    {
      int status;
      waitpid (pid, &status, 0);
      if (WIFEXITED (status))
	{
	  status = WEXITSTATUS (status);
	  MU_DEBUG1 (rp->debug, MU_DEBUG_TRACE0,
		     "Program finished with code %d.", status);
	  *pstatus = status;
	}
      else if (WIFSIGNALED (status))
	{
	  MU_DEBUG1 (rp->debug, MU_DEBUG_ERROR,
		     "Program terminated on signal %d.",
		     WTERMSIG (status));
	  return MU_ERR_PROCESS_SIGNALED;
	}
      else
	return MU_ERR_PROCESS_UNKNOWN_FAILURE;
    }
	
  return 0;
}
	    

int
_run_entry (void *item, void *data)
{
  int status = 0;
  struct _mu_acl_entry *ent = item;
  struct run_closure *rp = data;

  rp->idx++;

  if (mu_debug_check_level (rp->debug, MU_DEBUG_TRACE0))
    {
      const char *s = "undefined";
      mu_acl_action_to_string (ent->action, &s);
      __MU_DEBUG2 (rp->debug, MU_DEBUG_TRACE0, "%d:%s: ", rp->idx, s);
    }
  
  if (_acl_match (rp->debug, ent, rp->sa, rp->salen) == 0)
    {
      switch (ent->action)
	{
	case mu_acl_accept:
	  *rp->result = mu_acl_result_accept;
	  status = 1;
	  break;
      
	case mu_acl_deny:
	  *rp->result = mu_acl_result_deny;
	  status = 1;
	  break;
      
	case mu_acl_log:
	  {
	    char *s;
	    mu_debug_t dbg = NULL;
	    mu_diag_get_debug (&dbg);
	    if (ent->arg && expand_arg (ent->arg, rp, &s) == 0)
	      {
		mu_debug_printf (dbg, MU_DIAG_INFO, "%s\n", s);
		free (s);
	      }
	    else
	      {
		debug_sockaddr (dbg, MU_DIAG_INFO, rp->sa, rp->salen);
		mu_debug_printf (dbg, MU_DIAG_INFO, "\n");
	      }
	  }
	  break;
	  
	case mu_acl_exec:
	  spawn_prog (ent->arg, NULL, rp);
	  break;
	  
	case mu_acl_ifexec:
	  {
	    int prog_status;
	    int rc = spawn_prog (ent->arg, &prog_status, rp);
	    if (rc == 0)
	      {
		switch (prog_status)
		  {
		  case 0:
		    *rp->result = mu_acl_result_accept;
		    status = 1;
		    break;
		    
		  case 1:
		    *rp->result = mu_acl_result_deny;
		    status = 1;
		  }
	      }
	  }
	  break;
	}
    }
  
  if (mu_debug_check_level (rp->debug, MU_DEBUG_TRACE0))                     
    mu_debug_printf (rp->debug, MU_DEBUG_TRACE0, "\n");
  
  return status;
}

int
mu_acl_check_sockaddr (mu_acl_t acl, const struct sockaddr *sa, int salen,
		       mu_acl_result_t *pres)
{
  struct run_closure r;
  
  if (!acl)
    return EINVAL;

  r.sa = malloc (salen);
  if (!r.sa)
    return ENOMEM;
  memcpy (r.sa, sa, salen);
  if (prepare_sa (r.sa))
    {
      free (r.sa);
      return EINVAL;
    }
  r.salen = salen;
  
  if (mu_debug_check_level (acl->debug, MU_DEBUG_TRACE0))
    {
      __MU_DEBUG1 (acl->debug, MU_DEBUG_TRACE0, "%s", "Checking sockaddr ");
      debug_sockaddr (acl->debug, MU_DEBUG_TRACE0, r.sa, r.salen);
      mu_debug_printf (acl->debug, MU_DEBUG_TRACE0, "\n");
    }

  r.idx = 0;
  r.debug = acl->debug;
  r.result = pres;
  *r.result = mu_acl_result_undefined;
  mu_list_do (acl->aclist, _run_entry, &r);
  free (r.sa);
  return 0;
}
	       
int
mu_acl_check_inaddr (mu_acl_t acl, const struct in_addr *inp,
		     mu_acl_result_t *pres)
{
  struct sockaddr_in cs;
  int len = sizeof cs;

  cs.sin_family = AF_INET;
  cs.sin_addr = *inp;
  cs.sin_addr.s_addr = ntohl (cs.sin_addr.s_addr);
  return mu_acl_check_sockaddr (acl, (struct sockaddr *) &cs, len, pres);
}
  
int
mu_acl_check_ipv4 (mu_acl_t acl, unsigned int addr, mu_acl_result_t *pres)
{
  struct in_addr in;

  in.s_addr = addr;
  return mu_acl_check_inaddr (acl, &in, pres);
}

int
mu_acl_check_fd (mu_acl_t acl, int fd, mu_acl_result_t *pres)
{
  struct sockaddr_in cs;
  socklen_t len = sizeof cs;

  if (getpeername (fd, (struct sockaddr *) &cs, &len) < 0)
    {
      MU_DEBUG1 (acl->debug, MU_DEBUG_ERROR, 
		 "Cannot obtain IP address of client: %s",
		 mu_strerror (errno));
      return MU_ERR_FAILURE;
    }

  return mu_acl_check_sockaddr (acl, (struct sockaddr *) &cs, len, pres);
}

