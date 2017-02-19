/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
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

#include "imap4d.h"
#ifdef WITH_GSASL
# include <mailutils/gsasl.h>
#endif
#include "mailutils/libargp.h"
#include "tcpwrap.h"

mu_m_server_t server;
unsigned int idle_timeout;
int imap4d_transcript;

mu_mailbox_t mbox;              /* Current mailbox */
char *real_homedir;             /* Homedir as returned by user database */
char *imap4d_homedir;           /* Homedir as visible for the remote party */
char *modify_homedir;           /* Expression to produce imap4d_homedir */
int state = STATE_NONAUTH;      /* Current IMAP4 state */
struct mu_auth_data *auth_data; 

int login_disabled;             /* Disable LOGIN command */
int tls_required;               /* Require STARTTLS */
int create_home_dir;            /* Create home directory if it does not
				   exist */
int home_dir_mode = S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH;

int mailbox_mode[NS_MAX];

/* Saved command line. */
int imap4d_argc;                 
char **imap4d_argv;

enum imap4d_preauth preauth_mode;
char *preauth_program;
int preauth_only;
int ident_port;
char *ident_keyfile;
int ident_encrypt_only;

const char *program_version = "imap4d (" PACKAGE_STRING ")";
static char doc[] = N_("GNU imap4d -- the IMAP4D daemon.");

#define OPT_PREAUTH         259
#define OPT_FOREGROUND      260

static struct argp_option options[] = {
  { "foreground", OPT_FOREGROUND, 0, 0, N_("remain in foreground"), 0},
  { "inetd",  'i', 0, 0, N_("run in inetd mode"), 0},
  { "daemon", 'd', N_("NUMBER"), OPTION_ARG_OPTIONAL,
    N_("runs in daemon mode with a maximum of NUMBER children"), 0 },

  {"preauth", OPT_PREAUTH, NULL, 0,
   N_("start in preauth mode") },
  
  {NULL, 0, NULL, 0, NULL, 0}
};


static error_t imap4d_parse_opt (int key, char *arg,
				 struct argp_state *state);

static struct argp argp = {
  options,
  imap4d_parse_opt,
  NULL,
  doc,
  NULL,
  NULL, NULL
};

static const char *imap4d_capa[] = {
  "auth",
  "common",
  "debug",
  "mailbox",
  "locking",
  "logging",
  "license",
  NULL
};

static int imap4d_mainloop (int, FILE *, FILE *);

static error_t
imap4d_parse_opt (int key, char *arg, struct argp_state *state)
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
      
    case OPT_PREAUTH:
      preauth_mode = preauth_stdio;
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

static int
cb_mode (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  char *p;
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  home_dir_mode = strtoul (val->v.string, &p, 8);
  if (p[0] || (home_dir_mode & ~0777))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
                         _("invalid mode specification: %s"),
			 val->v.string);
  return 0;
}

int
parse_preauth_scheme (mu_debug_t debug, const char *scheme, mu_url_t url)
{
  int rc = 0;
  if (strcmp (scheme, "stdio") == 0)
    preauth_mode = preauth_stdio;
  else if (strcmp (scheme, "prog") == 0)
    {
      char *path;
      rc = mu_url_aget_path (url, &path);
      if (rc)
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       _("URL error: cannot get path: %s"),
			       mu_strerror (rc));
	  return 1;
	}
      preauth_program = path;
      preauth_mode = preauth_prog;
    }
  else if (strcmp (scheme, "ident") == 0)
    {
      struct servent *sp;
      long n;
      if (url && mu_url_get_port (url, &n) == 0)
	ident_port = (short) n;
      else if ((sp = getservbyname ("auth", "tcp")))
	ident_port = ntohs (sp->s_port);
      else
	ident_port = 113;
      preauth_mode = preauth_ident;
    }
  else
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("unknown preauth scheme"));
      rc = 1;
    }

  return rc;
}
      
/* preauth prog:///usr/sbin/progname
   preauth ident[://:port]
   preauth stdio
*/
static int
cb_preauth (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (strcmp (val->v.string, "stdio") == 0)
    preauth_mode = preauth_stdio;
  else if (strcmp (val->v.string, "ident") == 0)
    return parse_preauth_scheme (debug, val->v.string, NULL);
  else if (val->v.string[0] == '/')
    {
      preauth_program = xstrdup (val->v.string);
      preauth_mode = preauth_prog;
    }
  else
    {
      mu_url_t url;
      char *scheme;
      int rc = mu_url_create (&url, val->v.string);

      if (rc)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "mu_url_create", val->v.string, rc);
	  return 1;
	}
      rc = mu_url_parse (url);
      if (rc)
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       "%s: %s", val->v.string, mu_strerror (rc));
	  return 1;
	}

      rc = mu_url_aget_scheme (url, &scheme);
      if (rc)
	{
	  mu_url_destroy (&url);
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       _("URL error: %s"), mu_strerror (rc));
	  return 1;
	}

      rc = parse_preauth_scheme (debug, scheme, url);
      mu_url_destroy (&url);
      free (scheme);
      return rc;
    }
  return 0;
}

static int
cb_mailbox_mode (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  const char *p;
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (mu_parse_stream_perm_string ((int *)data, val->v.string, &p))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			 _("invalid mode string near %s"), p);
  return 0;
}

static struct mu_cfg_param imap4d_cfg_param[] = {
  { "homedir", mu_cfg_string, &modify_homedir, 0, NULL,
    N_("Modify home directory.") },
  { "personal-namespace", MU_CFG_LIST_OF(mu_cfg_string), &namespace[NS_PRIVATE],
    0, NULL, 
    N_("Set personal namespace.") },
  { "other-namespace", MU_CFG_LIST_OF(mu_cfg_string), &namespace[NS_OTHER],
    0, NULL, 
    N_("Set other users' namespace.") },
  { "shared-namespace", MU_CFG_LIST_OF(mu_cfg_string), &namespace[NS_SHARED],
    0, NULL,
    N_("Set shared namespace.") },
  { "other-mailbox-mode", mu_cfg_callback, &mailbox_mode[NS_OTHER], 0,
    cb_mailbox_mode,
    N_("File mode for mailboxes in other namespace.") },
  { "shared-mailbox-mode", mu_cfg_callback, &mailbox_mode[NS_SHARED], 0,
    cb_mailbox_mode,
    N_("File mode for mailboxes in shared namespace.") },
  { "login-disabled", mu_cfg_bool, &login_disabled, 0, NULL,
    N_("Disable LOGIN command.") },
  { "create-home-dir", mu_cfg_bool, &create_home_dir, 0, NULL,
    N_("If true, create non-existing user home directories.") },
  { "home-dir-mode", mu_cfg_callback, NULL, 0, cb_mode,
    N_("File mode for creating user home directories (octal)."),
    N_("mode") },
  { "tls-required", mu_cfg_bool, &tls_required, 0, NULL,
    N_("Always require STARTTLS before entering authentication phase.") },
  { "preauth", mu_cfg_callback, NULL, 0, cb_preauth,
    N_("Configure PREAUTH mode.  MODE is one of:\n"
       "  prog:///<full-program-name: string>\n"
       "  ident[://:<port: string-or-number>]\n"
       "  stdio"),
    N_("mode") },
  { "preauth-only", mu_cfg_bool, &preauth_only, 0, NULL,
    N_("Use only preauth mode.  If unable to setup it, disconnect "
       "immediately.") },
  { "ident-keyfile", mu_cfg_string, &ident_keyfile, 0, NULL,
    N_("Name of DES keyfile for decoding ecrypted ident responses.") },
  { "ident-encrypt-only", mu_cfg_bool, &ident_encrypt_only, 0, NULL,
    N_("Use only encrypted ident responses.") },
  { "id-fields", MU_CFG_LIST_OF(mu_cfg_string), &imap4d_id_list, 0, NULL,
    N_("List of fields to return in response to ID command.") },
  { ".server", mu_cfg_section, NULL, 0, NULL,
    N_("Server configuration.") },
  { "transcript", mu_cfg_bool, &imap4d_transcript, 0, NULL,
    N_("Set global transcript mode.") },
  TCP_WRAPPERS_CONFIG
  { NULL }
};

int
imap4d_session_setup0 ()
{
  real_homedir = mu_normalize_path (mu_strdup (auth_data->dir));
  if (imap4d_check_home_dir (real_homedir, auth_data->uid, auth_data->gid))
    return 1;

  if (modify_homedir)
    {
      int rc;
      mu_vartab_t vtab;
      char *expr = mu_tilde_expansion (modify_homedir, "/", real_homedir);

      mu_vartab_create (&vtab);
      mu_vartab_define (vtab, "user", auth_data->name, 0);
      mu_vartab_define (vtab, "home", real_homedir, 0);
      rc = mu_vartab_expand (vtab, expr, &imap4d_homedir);
      mu_vartab_destroy (&vtab);
      free (expr);
      if (rc)
	{
	  free (real_homedir);
	  mu_diag_funcall (MU_DIAG_ERROR, "mu_vartab_expand",
			   modify_homedir, rc);
	  return 1;
	}
    }
  else
    imap4d_homedir = strdup (real_homedir);

  if (strcmp (imap4d_homedir, real_homedir)
      && imap4d_check_home_dir (imap4d_homedir,
				auth_data->uid, auth_data->gid))
    {
      free (imap4d_homedir);
      free (real_homedir);
      return 1;
    }
  
  if (auth_data->change_uid)
    setuid (auth_data->uid);

  util_chdir (imap4d_homedir);
  namespace_init_session (imap4d_homedir);
  mu_diag_output (MU_DIAG_INFO,
		  _("user `%s' logged in (source: %s)"), auth_data->name,
		  auth_data->source);
  return 0;
}

int
imap4d_session_setup (char *username)
{
  auth_data = mu_get_auth_by_name (username);
  if (auth_data == NULL)
    {
      mu_diag_output (MU_DIAG_INFO, _("user `%s' nonexistent"), username);
      return 1;
    }
  return imap4d_session_setup0 ();
}

int
get_client_address (int fd, struct sockaddr_in *pcs)
{
  socklen_t len = sizeof *pcs;

  if (getpeername (fd, (struct sockaddr *) pcs, &len) < 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "getpeername", NULL, errno);
      return 1;
    }
  return 0;
}

void
imap4d_child_signal_setup (RETSIGTYPE (*handler) (int signo))
{
  static int sigtab[] = { SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGSTOP, SIGPIPE,
			  SIGABRT, SIGINT, SIGQUIT, SIGTERM, SIGHUP, SIGALRM };
  mu_set_signals (handler, sigtab, MU_ARRAY_SIZE (sigtab));
}

static int
imap4d_mainloop (int fd, FILE *infile, FILE *outfile)
{
  imap4d_tokbuf_t tokp;
  char *text;
  int debug_mode = isatty (fd);

  imap4d_child_signal_setup (imap4d_child_signal);
  util_setio (infile, outfile);

  if (imap4d_preauth_setup (fd) == 0)
    {
      if (debug_mode)
	{
	  mu_diag_output (MU_DIAG_INFO, _("started in debugging mode"));
	  text = "IMAP4rev1 Debugging mode";
	}
      else
	text = "IMAP4rev1";
    }
  else
    {
      util_flush_output ();
      return 0;
    }

  /* Greetings.  */
  util_out ((state == STATE_AUTH) ? RESP_PREAUTH : RESP_OK, "%s", text);
  util_flush_output ();

  tokp = imap4d_tokbuf_init ();
  while (1)
    {
      imap4d_readline (tokp);
      /* check for updates */
      imap4d_sync ();
      util_do_command (tokp);
      imap4d_sync ();
      util_flush_output ();
    }

  return 0;
}

int
imap4d_connection (int fd, struct sockaddr *sa, int salen, void *data,
		   mu_ip_server_t srv, time_t timeout, int transcript)
{
  idle_timeout = timeout;
  if (imap4d_transcript != transcript)
    imap4d_transcript = transcript;
  imap4d_mainloop (fd, fdopen (fd, "r"), fdopen (fd, "w"));
  return 0;
}

int
imap4d_check_home_dir (const char *dir, uid_t uid, gid_t gid)
{
  struct stat st;

  if (stat (dir, &st))
    {
      if (errno == ENOENT && create_home_dir)
	{
	  mode_t mode = umask (0);
	  int rc = mkdir (dir, home_dir_mode);
	  umask (mode);
	  if (rc)
	    {
	      mu_error ("Cannot create home directory `%s': %s",
			dir, mu_strerror (errno));
	      return 1;
	    }
	  if (chown (dir, uid, gid))
	    {
	      mu_error ("Cannot set owner for home directory `%s': %s",
			dir, mu_strerror (errno));
	      return 1;
	    }
	}
    }
  
  return 0;
}

int
main (int argc, char **argv)
{
  struct group *gr;
  int status = 0;
  static int sigtab[] = { SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGSTOP, SIGPIPE,
			  SIGABRT };

  imap4d_argc = argc;
  imap4d_argv = argv;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  state = STATE_NONAUTH;	/* Starting state in non-auth.  */

  MU_AUTH_REGISTER_ALL_MODULES ();
  /* Register the desired formats. */
  mu_register_local_mbox_formats ();
  
  imap4d_capability_init ();
#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif /* WITH_TLS */
#ifdef WITH_GSASL
  mu_gocs_register ("gsasl", mu_gsasl_module_init);
#endif
  mu_tcpwrapper_cfg_init ();
  mu_acl_cfg_init ();
  mu_m_server_cfg_init ();
  
  mu_argp_init (program_version, NULL);

  mu_m_server_create (&server, program_version);
  mu_m_server_set_conn (server, imap4d_connection);
  mu_m_server_set_prefork (server, mu_tcp_wrapper_prefork);
  mu_m_server_set_mode (server, MODE_INTERACTIVE);
  mu_m_server_set_max_children (server, 20);
  /* FIXME mu_m_server_set_pidfile (); */
  mu_m_server_set_default_port (server, 143);
  mu_m_server_set_timeout (server, 1800);  /* RFC2060: 30 minutes. */
  mu_m_server_set_strexit (server, mu_strexit);
  
  if (mu_app_init (&argp, imap4d_capa, imap4d_cfg_param, 
		   argc, argv, 0, NULL, server))
    exit (EX_CONFIG); /* FIXME: No way to discern from EX_USAGE? */

  if (login_disabled)
    imap4d_capability_add (IMAP_CAPA_LOGINDISABLED);
#ifdef WITH_TLS
  if (tls_required)
    imap4d_capability_add (IMAP_CAPA_XTLSREQUIRED);
#endif

  namespace_init ();
  
  auth_gssapi_init ();
  auth_gsasl_init ();

#ifdef USE_LIBPAM
  if (!mu_pam_service)
    mu_pam_service = "gnu-imap4d";
#endif

  if (mu_m_server_mode (server) == MODE_DAEMON)
    {
      /* Normal operation: */
      /* First we want our group to be mail so we can access the spool.  */
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
  mu_set_signals (imap4d_master_signal, sigtab, MU_ARRAY_SIZE (sigtab));

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
  starttls_init ();
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
      status = imap4d_mainloop (fileno (stdin), stdin, stdout);
    }

  if (status)
    mu_error (_("main loop status: %s"), mu_strerror (status));	  
  /* Close the syslog connection and exit.  */
  closelog ();

  return status ? EX_SOFTWARE : EX_OK;
}

