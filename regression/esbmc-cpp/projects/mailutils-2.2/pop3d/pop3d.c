/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

#include "pop3d.h"
#include "mailutils/pam.h"
#include "mailutils/libargp.h"
#include "tcpwrap.h"

mu_mailbox_t mbox;
int state;
char *username;
char *md5shared;

mu_m_server_t server;
unsigned int idle_timeout;
int pop3d_transcript;
int debug_mode;
int tls_required;

#ifdef WITH_TLS
int tls_available;
int tls_done;
#endif /* WITH_TLS */

int initial_state = AUTHORIZATION; 

/* Should all the messages be undeleted on startup */
int undelete_on_startup;
#ifdef ENABLE_LOGIN_DELAY
/* Minimum allowed delay between two successive logins */
time_t login_delay = 0;
char *login_stat_file = LOGIN_STAT_FILE;
#endif

unsigned expire = EXPIRE_NEVER; /* Expire messages after this number of days */
int expire_on_exit = 0;         /* Delete expired messages on exit */

static error_t pop3d_parse_opt  (int key, char *arg, struct argp_state *astate);

const char *program_version = "pop3d (" PACKAGE_STRING ")";
static char doc[] = N_("GNU pop3d -- the POP3 daemon.");

#define OPT_FOREGROUND      256

static struct argp_option options[] = {
#define GRP 0
  { "foreground", OPT_FOREGROUND, 0, 0, N_("remain in foreground"), GRP+1},
  { "inetd",  'i', 0, 0, N_("run in inetd mode"), GRP+1},
  { "daemon", 'd', N_("NUMBER"), OPTION_ARG_OPTIONAL,
    N_("runs in daemon mode with a maximum of NUMBER children"), GRP+1 },
#undef GRP

  {NULL, 0, NULL, 0, NULL, 0}
};

static int
cb_bulletin_source (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  set_bulletin_source (val->v.string); /* FIXME: Error reporting? */
  return 0;
}

#ifdef USE_DBM
static int
cb_bulletin_db (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  set_bulletin_db (val->v.string); /* FIXME: Error reporting? */
  return 0;
}
#endif

static struct mu_cfg_param pop3d_cfg_param[] = {
  { "undelete", mu_cfg_bool, &undelete_on_startup, 0, NULL,
    N_("On startup, clear deletion marks from all the messages.") },
  { "expire", mu_cfg_uint, &expire, 0, NULL,
    N_("Automatically expire read messages after the given number of days."),
    N_("days") },
  { "delete-expired", mu_cfg_bool, &expire_on_exit, 0, NULL,
    N_("Delete expired messages upon closing the mailbox.") },
#ifdef WITH_TLS
  { "tls-required", mu_cfg_bool, &tls_required, 0, NULL,
     N_("Always require STLS before entering authentication phase.") },
#endif
#ifdef ENABLE_LOGIN_DELAY
  { "login-delay", mu_cfg_time, &login_delay, 0, NULL,
    N_("Set the minimal allowed delay between two successive logins.") },
  { "stat-file", mu_cfg_string, &login_stat_file, 0, NULL,
    N_("Set the name of login statistics file (for login-delay).") },
#endif
  { "bulletin-source", mu_cfg_callback, NULL, 0, cb_bulletin_source,
    N_("Get bulletins from the specified mailbox."),
    N_("url") },
#ifdef USE_DBM
  { "bulletin-db", mu_cfg_callback, NULL, 0, cb_bulletin_db,
    N_("Set the bulletin database file name."),
    N_("file") },
#endif
  { ".server", mu_cfg_section, NULL, 0, NULL,
    N_("Server configuration.") },
  { "transcript", mu_cfg_bool, &pop3d_transcript, 0, NULL,
    N_("Set global transcript mode.") },
  TCP_WRAPPERS_CONFIG
  { NULL }
};
    
static struct argp argp = {
  options,
  pop3d_parse_opt,
  NULL,
  doc,
  NULL,
  NULL, NULL
};

static const char *pop3d_argp_capa[] = {
  "auth",
  "common",
  "debug",
  "mailbox",
  "locking",
  "logging",
  "license",
  NULL
};

static error_t
pop3d_parse_opt (int key, char *arg, struct argp_state *astate)
{
  static mu_list_t lst;
  
  switch (key)
    {
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

int
pop3d_get_client_address (int fd, struct sockaddr_in *pcs)
{
  mu_diag_output (MU_DIAG_INFO, _("incoming connection opened"));

  /* log information on the connecting client. */
  if (debug_mode)
    {
      mu_diag_output (MU_DIAG_INFO, _("started in debugging mode"));
      return 1;
    }
  else
    {
      socklen_t len = sizeof *pcs;
      if (getpeername (fd, (struct sockaddr*) pcs, &len) < 0)
	{
	  mu_diag_output (MU_DIAG_ERROR,
			  _("cannot obtain IP address of client: %s"),
			  strerror (errno));
	  return 1;
	}
    }
  return 0;
}

/* The main part of the daemon. This function reads input from the client and
   executes the proper functions. Also handles the bulk of error reporting.
   Arguments:
      fd        --  socket descriptor (for diagnostics)
      infile    --  input stream
      outfile   --  output stream */
int
pop3d_mainloop (int fd, FILE *infile, FILE *outfile)
{
  int status = OK;
  char buffer[512];
  static int sigtab[] = { SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGSTOP, SIGPIPE,
			  SIGABRT, SIGINT, SIGQUIT, SIGTERM, SIGHUP, SIGALRM };

  mu_set_signals (pop3d_child_signal, sigtab, MU_ARRAY_SIZE (sigtab));

  pop3d_setio (infile, outfile);

  state = initial_state;

  /* Prepare the shared secret for APOP.  */
  {
    char *local_hostname;
    local_hostname = mu_alloc (MAXHOSTNAMELEN + 1);

    /* Get our canonical hostname. */
    {
      struct hostent *htbuf;
      gethostname (local_hostname, MAXHOSTNAMELEN);
      htbuf = gethostbyname (local_hostname);
      if (htbuf)
	{
	  free (local_hostname);
	  local_hostname = strdup (htbuf->h_name);
	}
    }

    md5shared = mu_alloc (strlen (local_hostname) + 51);

    snprintf (md5shared, strlen (local_hostname) + 50, "<%u.%u@%s>", getpid (),
	      (unsigned)time (NULL), local_hostname);
    free (local_hostname);
  }

  /* Lets boogie.  */
  pop3d_outf ("+OK POP3 Ready %s\r\n", md5shared);

  while (state != UPDATE && state != ABORT)
    {
      char *buf;
      char *arg, *cmd;
      pop3d_command_handler_t handler;
      
      pop3d_flush_output ();
      status = OK;
      buf = pop3d_readline (buffer, sizeof (buffer));
      pop3d_parse_command (buf, &cmd, &arg);

      /* The mailbox size needs to be check to make sure that we are in
	 sync.  Some other applications may not respect the *.lock or
	 the lock may be stale because downloading on slow modem.
	 We rely on the size of the mailbox for the check and bail if out
	 of sync.  */
      if (state == TRANSACTION && !mu_mailbox_is_updated (mbox))
	{
	  static mu_off_t mailbox_size;
	  mu_off_t newsize = 0;
	  mu_mailbox_get_size (mbox, &newsize);
	  /* Did we shrink?  First time save the size.  */
	  if (!mailbox_size)
	    mailbox_size = newsize;
	  else if (newsize < mailbox_size) /* FIXME: Should it be a != ? */
	    pop3d_abquit (ERR_MBOX_SYNC); /* Out of sync, Bail out.  */
	}

      /* Refresh the Lock.  */
      pop3d_touchlock ();

      if (strlen (arg) > POP_MAXCMDLEN || strlen (cmd) > POP_MAXCMDLEN)
	status = ERR_TOO_LONG;
      else if (strlen (cmd) > 4)
	status = ERR_BAD_CMD;
      else if ((handler = pop3d_find_command (cmd)) != NULL)
	status = handler (arg);
      else
	status = ERR_BAD_CMD;

      if (status != OK)
	pop3d_outf ("-ERR %s\r\n", pop3d_error_string (status));
    }

  pop3d_bye ();

  return status;
}

int
pop3d_connection (int fd, struct sockaddr *sa, int salen, void *data,
		  mu_ip_server_t srv, time_t timeout, int transcript)
{
  idle_timeout = timeout;
  if (pop3d_transcript != transcript)
    pop3d_transcript = transcript;
  pop3d_mainloop (fd, fdopen (fd, "r"), fdopen (fd, "w"));
  return 0;
}

static void
pop3d_alloc_die ()
{
  pop3d_abquit (ERR_NO_MEM);
}

int
main (int argc, char **argv)
{
  struct group *gr;
  int status = OK;
  static int sigtab[] = { SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGSTOP, SIGPIPE,
			  SIGABRT };

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  MU_AUTH_REGISTER_ALL_MODULES();
  /* Register the desired formats.  */
  mu_register_local_mbox_formats ();

#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif /* WITH_TLS */
  mu_tcpwrapper_cfg_init ();
  mu_acl_cfg_init ();
  mu_m_server_cfg_init ();
  
  mu_argp_init (program_version, NULL);
  	
  mu_m_server_create (&server, program_version);
  mu_m_server_set_conn (server, pop3d_connection);
  mu_m_server_set_prefork (server, mu_tcp_wrapper_prefork);
  mu_m_server_set_mode (server, MODE_INTERACTIVE);
  mu_m_server_set_max_children (server, 20);
  /* FIXME mu_m_server_set_pidfile (); */
  mu_m_server_set_default_port (server, 110);
  mu_m_server_set_timeout (server, 600);
  mu_m_server_set_strexit (server, mu_strexit);

  mu_alloc_die_hook = pop3d_alloc_die;
  
  if (mu_app_init (&argp, pop3d_argp_capa, pop3d_cfg_param, 
		   argc, argv, 0, NULL, server))
    exit (EX_CONFIG); /* FIXME: No way to discern from EX_USAGE? */

  if (tls_required)
    initial_state = INITIAL;
  
  if (expire == 0)
    expire_on_exit = 1;

#ifdef USE_LIBPAM
  if (!mu_pam_service)
    mu_pam_service = "gnu-pop3d";
#endif

  if (mu_m_server_mode (server) == MODE_INTERACTIVE && isatty (0))
    {
      /* If input is a tty, switch to debug mode */
      debug_mode = 1;
    }
  else
    {
      errno = 0;
      gr = getgrnam ("mail");
      if (gr == NULL)
	{
	  if (errno == 0 || errno == ENOENT)
            {
               mu_error (_("%s: no such group"), "mail");
               exit (EX_CONFIG);
            }
          else
            {
	      mu_diag_funcall (MU_DIAG_ERROR, "getgrnam", "mail", errno);
	      exit (EX_OSERR);
            }
	}

      if (setgid (gr->gr_gid) == -1)
	{
	  mu_error (_("error setting mail group: %s"), mu_strerror (errno));
	  exit (EX_OSERR);
	}
    }

  /* Set the signal handlers.  */
  mu_set_signals (pop3d_master_signal, sigtab, MU_ARRAY_SIZE (sigtab));

  /* Set up for syslog.  */
  openlog (MU_LOG_TAG (), LOG_PID, mu_log_facility);
  /* Redirect any stdout error from the library to syslog, they
     should not go to the client.  */
  {
    mu_debug_t debug;

    mu_diag_get_debug (&debug);
    mu_debug_set_print (debug, mu_diag_syslog_printer, NULL);
    
    mu_debug_default_printer = mu_debug_syslog_printer;
  }
  
  umask (S_IROTH | S_IWOTH | S_IXOTH);	/* 007 */

  /* Check TLS environment, i.e. cert and key files */
#ifdef WITH_TLS
  tls_available = mu_check_tls_environment ();
  if (tls_available)
    {
      tls_available = mu_init_tls_libs ();
      if (tls_available)
	enable_stls ();
    }
#endif /* WITH_TLS */

  /* Actually run the daemon.  */
  if (mu_m_server_mode (server) == MODE_DAEMON)
    {
      mu_m_server_begin (server);
      status = mu_m_server_run (server);
      mu_m_server_end (server);
      mu_m_server_destroy (&server);
    }
  else
    {
      /* Make sure we are in the root directory.  */
      chdir ("/");
      status = pop3d_mainloop (fileno (stdin), stdin, stdout);
    }
  
  if (status)
    mu_error (_("main loop status: %s"), mu_strerror (status));	  
  /* Close the syslog connection and exit.  */
  closelog ();
  return status ? EX_SOFTWARE : EX_OK;
}

