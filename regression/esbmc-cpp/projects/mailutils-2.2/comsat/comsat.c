/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "comsat.h"
#define MU_CFG_COMPATIBILITY /* This source uses deprecated cfg interfaces */
#include "mailutils/libcfg.h"
#include "mailutils/libargp.h"

#ifndef PATH_DEV
# define PATH_DEV "/dev"
#endif
#ifndef PATH_TTY_PFX
# define PATH_TTY_PFX PATH_DEV
#endif

#ifdef HAVE_UTMP_H
# include <utmp.h>
#endif

#ifndef HAVE_GETUTENT_CALLS
extern void setutent (void);
extern struct utmp *getutent (void);
#endif

#ifdef UTMPX
# ifdef HAVE_UTMPX_H
#  include <utmpx.h>
# endif
typedef struct utmpx UTMP;
# define SETUTENT() setutxent()
# define GETUTENT() getutxent()
# define ENDUTENT() endutxent()
#else
typedef struct utmp UTMP;
# define SETUTENT() setutent()
# define GETUTENT() getutent()
# define ENDUTENT() endutent()
#endif

#define MAX_TTY_SIZE (sizeof (PATH_TTY_PFX) + sizeof (((UTMP*)0)->ut_line))

const char *program_version = "comsatd (" PACKAGE_STRING ")";
static char doc[] = N_("GNU comsatd -- the Comsat daemon.");
static char args_doc[] = N_("\n--test MBOX-URL MSG-QID");

#define OPT_FOREGROUND 256

static struct argp_option options[] = 
{
  { "config", 'c', N_("FILE"), OPTION_HIDDEN, "", 0 },
  { "convert-config", 'C', N_("FILE"), 0,
    N_("convert the configuration FILE to new format"), 0 },
  { "test", 't', NULL, 0, N_("run in test mode"), 0 },
  { "foreground", OPT_FOREGROUND, 0, 0, N_("remain in foreground"), 0},
  { "inetd",  'i', 0, 0, N_("run in inetd mode"), 0 },
  { "daemon", 'd', N_("NUMBER"), OPTION_ARG_OPTIONAL,
    N_("runs in daemon mode with a maximum of NUMBER children"), 0 },
  { NULL, 0, NULL, 0, NULL, 0 }
};

static error_t comsatd_parse_opt (int key, char *arg,
				  struct argp_state *state);

static struct argp argp = {
  options,
  comsatd_parse_opt,
  args_doc, 
  doc,
  NULL,
  NULL, NULL
};

static const char *comsat_argp_capa[] = {
  "common",
  "debug",
  "logging",
  "mailbox",
  "locking",
  "license",
  NULL
};

#define SUCCESS 0
#define NOT_HERE 1
#define PERMISSION_DENIED 2

#ifndef MAXHOSTNAMELEN
# define MAXHOSTNAMELEN 64
#endif

int maxlines = 5;
char hostname[MAXHOSTNAMELEN];
const char *username;
int require_tty;
mu_m_server_t server;

static void comsat_init (void);
static int comsat_main (int fd);
static void notify_user (const char *user, const char *device,
			 const char *path, mu_message_qid_t qid);
static int find_user (const char *name, char *tty);
static char *mailbox_path (const char *user);
static int change_user (const char *user);

static int reload = 0;
int test_mode;

struct mu_cfg_param comsat_cfg_param[] = {
  { "allow-biffrc", mu_cfg_bool, &allow_biffrc, 0, NULL,
    N_("Read .biffrc file from the user home directory.") },
  { "require-tty", mu_cfg_bool, &require_tty, 0, NULL,
    N_("Notify only if the user is logged on one of the ttys.") },
  { "max-lines", mu_cfg_int, &maxlines, 0, NULL,
    N_("Maximum number of message body lines to be output.") },
  { "max-requests", mu_cfg_uint, &maxrequests, 0, NULL,
    N_("Maximum number of incoming requests per request control interval.") },
  { "request-control-interval", mu_cfg_time, &request_control_interval,
    0, NULL,
    N_("Set control interval.") },
  { "overflow-control-interval", mu_cfg_time, &overflow_control_interval,
    0, NULL,
    N_("Set overflow control interval.") },
  { "overflow-delay-time", mu_cfg_time, &overflow_delay_time,
    0, NULL,
    N_("Time to sleep after the first overflow occurs.") },
  { ".server", mu_cfg_section, NULL, 0, NULL,
    N_("Server configuration.") },
  { NULL }
};

static error_t
comsatd_parse_opt (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
    case 'c':
      {
	char *cfg;
	int fd;
	FILE *fp;

	mu_diag_output (MU_DIAG_WARNING,
_("The old configuration file format and the --config command\n"
  "line option are deprecated and will be removed in the future\n"
  "release. Please use --convert-config option to convert your\n"
  "settings to the new format."));
	/* FIXME: Refer to the docs */
	
	fd = mu_tempfile (NULL, &cfg);
	fp = fdopen (fd, "w");
	convert_config (arg, fp);
	fclose (fp);
	mu_get_config (cfg, mu_program_name, comsat_cfg_param, 0, NULL);
	unlink (cfg);
	free (cfg);
      }
      break;
      
    case 'C':
      convert_config (arg, stdout);
      exit (0);

    case 'd':
      mu_argp_node_list_new (lst, "mode", "daemon");
      if (arg)
	mu_argp_node_list_new (lst, "max-children", arg);
      break;

    case 'i':
      mu_argp_node_list_new (lst, "mode", "inetd");
      break;

    case OPT_FOREGROUND:
      mu_argp_node_list_new (lst, "foreground", "yes");
      break;

    case 't':
      test_mode = 1;
      break;
      
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, NULL, NULL);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static RETSIGTYPE
sig_hup (int sig)
{
  mu_m_server_stop (1);
  reload = 1;
}

void
comsat_init ()
{
  /* Register mailbox formats */
  mu_register_all_mbox_formats ();

  gethostname (hostname, sizeof hostname);

  /* Set signal handlers */
  signal (SIGTTOU, SIG_IGN);
  signal (SIGCHLD, SIG_IGN);
  signal (SIGHUP, SIG_IGN);	/* Ignore SIGHUP.  */
}

int allow_biffrc = 1;            /* Allow per-user biffrc files */
unsigned maxrequests = 16;       /* Maximum number of request allowed per
			            control interval */
time_t request_control_interval = 10;  /* Request control interval */
time_t overflow_control_interval = 10; /* Overflow control interval */
time_t overflow_delay_time = 5;

void
comsat_process (char *buffer, size_t rdlen)
{
  char tty[MAX_TTY_SIZE];
  char *p;
  char *path = NULL;
  mu_message_qid_t qid;

  /* Parse the buffer */
  p = strchr (buffer, '@');
  if (!p)
    {
      mu_diag_output (MU_DIAG_ERROR, _("malformed input: %s"), buffer);
      return;
    }
  *p++ = 0;

  qid = p;
  p = strchr (qid, ':');
  if (p)
    {
      *p++ = 0;
      path = p;
    }
    
  if (find_user (buffer, tty) != SUCCESS)
    {
      if (require_tty)
	return;
      strcpy (tty, "/dev/null");
    }

  /* Child: do actual I/O */
  notify_user (buffer, tty, path, qid);
}

int
comsat_main (int fd)
{
  int rdlen;
  socklen_t len;
  struct sockaddr fromaddr;
  char buffer[216]; /*FIXME: Arbitrary size */

  len = sizeof fromaddr;
  rdlen = recvfrom (fd, buffer, sizeof buffer, 0, &fromaddr, &len);
  if (rdlen <= 0)
    {
      if (errno == EINTR)
	return 0;
      mu_diag_output (MU_DIAG_ERROR, "recvfrom: %m");
      return 1;
    }
  buffer[rdlen] = 0;
  
  if (mu_m_server_check_acl (server, &fromaddr, len))
    return 0;

  comsat_process (buffer, rdlen);
  return 0;
}

static time_t last_request_time;    /* Timestamp of the last received
				       request */
static unsigned reqcount = 0;       /* Number of request received in the
				       current control interval */
static time_t last_overflow_time;   /* Timestamp of last overflow */
static unsigned overflow_count = 0; /* Number of overflows detected during
				       the current interval */

int
comsat_prefork (int fd, void *data, struct sockaddr *s, int size)
{
  int retval = 0;
  time_t now;
  
  /* Control the request flow */
  if (maxrequests != 0)
    {
      now = time (NULL);
      if (reqcount > maxrequests)
	{
	  unsigned delay;

	  delay = overflow_delay_time << (overflow_count + 1);
	  mu_diag_output (MU_DIAG_NOTICE,
			 ngettext ("too many requests: pausing for %u second",
				   "too many requests: pausing for %u seconds",
				    delay),
			  delay);
	  /* FIXME: drain the socket? */
	  sleep (delay);
	  reqcount = 0;
	  if (now - last_overflow_time <= overflow_control_interval)
	    {
	      if ((overflow_delay_time << (overflow_count + 2)) >
		  overflow_delay_time)
		++overflow_count;
	    }
	  else
	    overflow_count = 0;
	  last_overflow_time = time (NULL);
	  retval = 1;
	}

      if (now - last_request_time <= request_control_interval)
	reqcount++;
      else
	{
	  last_request_time = now;
	  reqcount = 1;
	}
    }
  return retval;
}

int
comsat_connection (int fd, struct sockaddr *sa, int salen,
		   void *data, mu_ip_server_t srv,
		   time_t to, int transcript)
{
  char *buffer;
  size_t rdlen, size;

  if (mu_udp_server_get_rdata (srv, &buffer, &rdlen))
    return 0;
  if (transcript)
    {
      char *p = mu_sockaddr_to_astr (sa, salen);
      mu_diag_output (MU_DIAG_INFO,
		      ngettext ("received %d byte from %s",
				"received %d bytes from %s", rdlen),
		      rdlen, p);
      mu_diag_output (MU_DIAG_INFO, "string: %s", buffer);
      free (p);
    }
  mu_udp_server_get_bufsize (srv, &size);
  if (size < rdlen + 1)
    {
      int rc = mu_udp_server_set_bufsize (srv, rdlen + 1);
      if (rc)
	{
	  mu_error (_("cannot resize buffer: %s"), mu_strerror (rc));
	  return 0;
	}
    }
  buffer[rdlen] = 0;
  comsat_process (buffer, rdlen);
  return 0;
}

static const char *
get_newline_str (FILE *fp)
{
#if defined(OPOST) && defined(ONLCR)
  struct termios tbuf;

  tcgetattr (fileno (fp), &tbuf);
  if ((tbuf.c_oflag & OPOST) && (tbuf.c_oflag & ONLCR))
    return "\n";
  else
    return "\r\n";
#else
  return "\r\n"; /* Just in case */
#endif
}

/* NOTE: Do not bother to free allocated memory, as the program exits
   immediately after executing this */
static void
notify_user (const char *user, const char *device, const char *path,
	     mu_message_qid_t qid)
{
  FILE *fp;
  const char *cr;
  mu_mailbox_t mbox = NULL;
  mu_message_t msg;
  int status;

  if (change_user (user))
    return;
  if ((fp = fopen (device, "w")) == NULL)
    {
      mu_error (_("cannot open device %s: %s"), device, mu_strerror (errno));
      return;
    }

  cr = get_newline_str (fp);

  if (!path)
    {
      path = mailbox_path (user);
      if (!path)
	return;
    }

  if ((status = mu_mailbox_create (&mbox, path)) != 0
      || (status = mu_mailbox_open (mbox, MU_STREAM_READ|MU_STREAM_QACCESS)) != 0)
    {
      mu_error (_("cannot open mailbox %s: %s"),
	      path, mu_strerror (status));
      return;
    }

  status = mu_mailbox_quick_get_message (mbox, qid, &msg);
  if (status)
    {
      mu_error (_("cannot get message (mailbox %s, qid %s): %s"),
		path, qid, mu_strerror (status));
      return; /* FIXME: Notify the user, anyway */
    }

  run_user_action (fp, cr, msg);
  fclose (fp);
}

/* Search utmp for the local user */
static int
find_user (const char *name, char *tty)
{
  UTMP *uptr;
  int status;
  struct stat statb;
  char ftty[MAX_TTY_SIZE];
  time_t last_time = 0;

  status = NOT_HERE;
  sprintf (ftty, "%s/", PATH_TTY_PFX);

  SETUTENT ();

  while ((uptr = GETUTENT ()) != NULL)
    {
#ifdef USER_PROCESS
      if (uptr->ut_type != USER_PROCESS)
	continue;
#endif
      if (!strncmp (uptr->ut_name, name, sizeof(uptr->ut_name)))
	{
	  /* no particular tty was requested */
	  strncpy (ftty + sizeof(PATH_DEV),
		   uptr->ut_line,
		   sizeof (ftty) - sizeof (PATH_DEV) - 2);
	  ftty[sizeof (ftty) - 1] = 0;

	  mu_normalize_path (ftty);
	  if (strncmp (ftty, PATH_TTY_PFX, strlen (PATH_TTY_PFX)))
	    {
	      /* An attempt to break security... */
	      mu_diag_output (MU_DIAG_ALERT,
			      _("bad line name in utmp record: %s"), ftty);
	      return NOT_HERE;
	    }

	  if (stat (ftty, &statb) == 0)
	    {
	      if (!S_ISCHR (statb.st_mode))
		{
		  mu_diag_output (MU_DIAG_ALERT,
				  _("not a character device: %s"), ftty);
		  return NOT_HERE;
		}

	      if (!(statb.st_mode & S_IEXEC))
		{
		  if (status != SUCCESS)
		    status = PERMISSION_DENIED;
		  continue;
		}
	      if (statb.st_atime > last_time)
		{
		  last_time = statb.st_atime;
		  strcpy(tty, ftty);
		  status = SUCCESS;
		}
	      continue;
	    }
	}
    }

  ENDUTENT ();
  return status;
}

int
change_user (const char *user)
{
  struct passwd *pw;

  pw = getpwnam (user);
  if (!pw)
    {
      mu_diag_output (MU_DIAG_CRIT, _("no such user: %s"), user);
      return 1;
    }

  setgid (pw->pw_gid);
  setuid (pw->pw_uid);
  chdir (pw->pw_dir);
  username = user;
  return 0;
}

char *
mailbox_path (const char *user)
{
  struct mu_auth_data *auth;
  char *mailbox_name;

  auth = mu_get_auth_by_name (user);

  if (!auth)
    {
      mu_diag_output (MU_DIAG_ALERT, _("user nonexistent: %s"), user);
      return NULL;
    }

  mailbox_name = strdup (auth->mailbox);
  mu_auth_data_free (auth);
  return mailbox_name;
}


int
main (int argc, char **argv)
{
  int c;
  int ind;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mu_argp_init (program_version, NULL);
  comsat_init ();
  mu_acl_cfg_init ();
  mu_m_server_cfg_init ();
  mu_m_server_create (&server, program_version);
  mu_m_server_set_type (server, MU_IP_UDP);
  mu_m_server_set_conn (server, comsat_connection);
  mu_m_server_set_prefork (server, comsat_prefork);
  mu_m_server_set_mode (server, MODE_INTERACTIVE);
  mu_m_server_set_max_children (server, 20);
  /* FIXME mu_m_server_set_pidfile (); */
  mu_m_server_set_default_port (server, 512);
  /* FIXME: timeout is not needed. How to disable it? */
  
  if (mu_app_init (&argp, comsat_argp_capa, comsat_cfg_param, argc, argv, 0,
		   &ind, server))
    exit (1);

  if (test_mode)
    {
      char *user;
      
      argc -= ind;
      argv += ind;
  
      if (argc < 2 || argc > 2)
	{
	  mu_error (_("mailbox URL and message QID are required in test mode"));
	  exit (EXIT_FAILURE);
	}

      user = getenv ("LOGNAME");
      if (!user)
	{
	  user = getenv ("USER");
	  if (!user)
	    {
	      struct passwd *pw = getpwuid (getuid ());
	      if (!pw)
		{
		  mu_error (_("cannot determine user name"));
		  exit (EXIT_FAILURE);
		}
	      user = pw->pw_name;
	    }
	}
		  
      notify_user (user, "/dev/tty", argv[0], argv[1]);
      exit (0);
    }
  
  /* Set up error messaging  */
  openlog (MU_LOG_TAG (), LOG_PID, mu_log_facility);

  {
    mu_debug_t debug;

    mu_diag_get_debug (&debug);
    mu_debug_set_print (debug, mu_diag_syslog_printer, NULL);

    mu_debug_default_printer = mu_debug_syslog_printer;
  }

  if (mu_m_server_mode (server) == MODE_DAEMON)
    {
      if (argv[0][0] != '/')
	mu_diag_output (MU_DIAG_NOTICE,
			_("program name is not absolute; reloading will not "
			  "be possible"));
      else
	{
	  sigset_t set;

	  mu_m_server_get_sigset (server, &set);
	  sigdelset (&set, SIGHUP);
	  mu_m_server_set_sigset (server, &set);
	  signal (SIGHUP, sig_hup);
	}
      
      mu_m_server_begin (server);
      c = mu_m_server_run (server);
      mu_m_server_end (server);
      mu_m_server_destroy (&server);
      if (reload)
	{
	  mu_diag_output (MU_DIAG_NOTICE, _("restarting"));
	  execvp (argv[0], argv);
	}
    }
  else
    {
      chdir ("/");
      c = comsat_main (0);
    }
  
  return c != 0;
}

