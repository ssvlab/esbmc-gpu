/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

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

#include "maidag.h"

int multiple_delivery;     /* Don't return errors when delivering to multiple
			      recipients */
int ex_quota_tempfail;     /* Return temporary failure if mailbox quota is
			      exceeded. If this variable is not set, maidag
			      will return "service unavailable" */
int exit_code = EX_OK;     /* Exit code to be used */
uid_t current_uid;         /* Current user id */

char *quotadbname = NULL;  /* Name of mailbox quota database */
char *quota_query = NULL;  /* SQL query to retrieve mailbox quota */

char *sender_address = NULL;       

maidag_script_fun script_handler;

mu_list_t script_list;

char *forward_file = NULL;
int forward_file_checks = FWD_ALL;

int log_to_stderr = -1;

/* Debuggig options */
int debug_level;           /* General debugging level */ 
int sieve_debug_flags;     /* Sieve debugging flags */
int sieve_enable_log;      /* Enables logging of executed Sieve actions */
char *message_id_header;   /* Use the value of this header as message
			      identifier when logging Sieve actions */

/* For LMTP mode */
mu_m_server_t server;
int lmtp_mode;
int url_option;
char *lmtp_url_string;
int reuse_lmtp_address = 1;

const char *program_version = "maidag (" PACKAGE_STRING ")";
static char doc[] =
N_("GNU maidag -- the mail delivery agent.")
"\v"
N_("Debug flags are:\n\
  g - guimb stack traces\n\
  t - sieve trace (MU_SIEVE_DEBUG_TRACE)\n\
  i - sieve instructions trace (MU_SIEVE_DEBUG_INSTR)\n\
  l - sieve action logs\n\
  0-9 - Set maidag debugging level\n");

static char args_doc[] = N_("[recipient...]");

#define STDERR_OPTION 256
#define MESSAGE_ID_HEADER_OPTION 257
#define LMTP_OPTION 258
#define FOREGROUND_OPTION 260
#define URL_OPTION 261

static struct argp_option options[] = 
{
#define GRID 0
 { NULL, 0, NULL, 0,
   N_("General options"), GRID },
      
  { "foreground", FOREGROUND_OPTION, 0, 0, N_("remain in foreground"),
    GRID + 1 },
  { "inetd",  'i', 0, 0, N_("run in inetd mode"), GRID + 1 },
  { "daemon", 'd', N_("NUMBER"), OPTION_ARG_OPTIONAL,
    N_("runs in daemon mode with a maximum of NUMBER children"), GRID + 1 },
  { "url", URL_OPTION, 0, 0, N_("deliver to given URLs"), GRID + 1 },
  { "from", 'f', N_("EMAIL"), 0,
    N_("specify the sender's name"), GRID + 1 },
  { NULL, 'r', NULL, OPTION_ALIAS, NULL },
  { "lmtp", LMTP_OPTION, N_("URL"), OPTION_ARG_OPTIONAL,
    N_("operate in LMTP mode"), GRID + 1 },
  { "debug", 'x', N_("FLAGS"), 0,
    N_("enable debugging"), GRID + 1 },
  { "stderr", STDERR_OPTION, NULL, 0,
    N_("log to standard error"), GRID + 1 },
#undef GRID

#define GRID 2
 { NULL, 0, NULL, 0,
   N_("Scripting options"), GRID },
 
  { "language", 'l', N_("STRING"), 0,
    N_("define scripting language for the next --script option"),
    GRID + 1 },
  { "script", 's', N_("PATTERN"), 0,
    N_("set name pattern for user-defined mail filter"), GRID + 1 },
  { "message-id-header", MESSAGE_ID_HEADER_OPTION, N_("STRING"), 0,
    N_("use this header to identify messages when logging Sieve actions"),
    GRID + 1 },
#undef GRID
  { NULL,      0, NULL, 0, NULL, 0 }
};

static error_t parse_opt (int key, char *arg, struct argp_state *state);

static struct argp argp = {
  options,
  parse_opt,
  args_doc, 
  doc,
  NULL,
  NULL, NULL
};

static const char *maidag_argp_capa[] = {
  "auth",
  "common",
  "debug",
  "license",
  "logging",
  "mailbox",
  "locking",
  "mailer",
  NULL
};

#define D_DEFAULT "9,s"

static void
set_debug_flags (mu_debug_t debug, const char *arg)
{
  while (*arg)
    {
      if (mu_isdigit (*arg))
	debug_level = strtoul (arg, (char**)&arg, 10);
      else
	for (; *arg && *arg != ','; arg++)
	  {
	    switch (*arg)
	      {
	      case 'g':
#ifdef WITH_GUILE
		debug_guile = 1;
#endif
		break;

	      case 't':
		sieve_debug_flags |= MU_SIEVE_DEBUG_TRACE;
		break;
	  
	      case 'i':
		sieve_debug_flags |= MU_SIEVE_DEBUG_INSTR;
		break;
	  
	      case 'l':
		sieve_enable_log = 1;
		break;
	  
	      default:
		mu_cfg_format_error (debug, MU_DEBUG_ERROR,
				     _("%c is not a valid debug flag"), *arg);
		break;
	      }
	  }
      if (*arg == ',')
	arg++;
      else if (*arg)
	mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			     _("expected comma, but found %c"), *arg);
    }
}

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
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

    case FOREGROUND_OPTION:
      mu_argp_node_list_new (lst, "foreground", "yes");
      break;
      
    case MESSAGE_ID_HEADER_OPTION:
      mu_argp_node_list_new (lst, "message-id-header", arg);
      break;

    case LMTP_OPTION:
      mu_argp_node_list_new (lst, "lmtp", "yes");
      if (arg)
	mu_argp_node_list_new (lst, "listen", arg);
      break;

    case 'r':
    case 'f':
      if (sender_address != NULL)
	argp_error (state, _("multiple --from options"));
      sender_address = arg;
      break;
      
    case 'l':
      script_handler = script_lang_handler (arg);
      if (!script_handler)
	argp_error (state, _("unknown or unsupported language: %s"),
		    arg);
      break;
      
    case 's':
      switch (script_register (arg))
	{
	case 0:
	  break;

	case EINVAL:
	  argp_error (state, _("%s has unknown file suffix"), arg);
	  break;

	default:
	  argp_error (state, _("error registering script"));
	}
      break;
      
    case 'x':
      mu_argp_node_list_new (lst, "debug", arg ? arg : D_DEFAULT);
      break;

    case STDERR_OPTION:
      mu_argp_node_list_new (lst, "stderr", "yes");
      break;

    case URL_OPTION:
      url_option = 1;
      break;
      
    case ARGP_KEY_INIT:
      mu_argp_node_list_init (&lst);
      break;
      
    case ARGP_KEY_FINI:
      mu_argp_node_list_finish (lst, NULL, NULL);
      break;
      
    case ARGP_KEY_ERROR:
      exit (EX_USAGE);

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}



static int
cb_debug (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  set_debug_flags (debug, val->v.string);
  return 0;
}

static int
cb2_group (mu_debug_t debug, const char *gname, void *data)
{
  mu_list_t *plist = data;
  struct group *group;

  if (!*plist)
    mu_list_create (plist);
  group = getgrnam (gname);
  if (!group)
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("unknown group: %s"), gname);
  else
    mu_list_append (*plist, (void*)group->gr_gid);
  return 0;
}
  
static int
cb_group (mu_debug_t debug, void *data, mu_config_value_t *arg)
{
  return mu_cfg_string_value_cb (debug, arg, cb2_group, data);
}

static struct mu_kwd forward_checks[] = {
  { "all", FWD_ALL },
  { "groupwritablefile", FWD_IWGRP },
  { "file_iwgrp", FWD_IWGRP },
  { "worldwritablefile", FWD_IWOTH },
  { "file_iwoth", FWD_IWOTH },
  { "linkedfileinwritabledir", FWD_LINK },
  { "link", FWD_LINK },
  { "fileingroupwritabledir", FWD_DIR_IWGRP },
  { "dir_iwgrp", FWD_DIR_IWGRP },
  { "fileinworldwritabledir", FWD_DIR_IWOTH },
  { "dir_iwoth", FWD_DIR_IWOTH },
  { NULL }
};

static int
cb2_forward_file_checks (mu_debug_t debug, const char *name, void *data)
{
  int negate = 0;
  const char *str;
  int val;
  
  if (strlen (name) > 2 && mu_c_strncasecmp (name, "no", 2) == 0)
    {
      negate = 1;
      str = name + 2;
    }
  else
    str = name;

  if (mu_kwd_xlat_name_ci (forward_checks, str, &val))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("unknown keyword: %s"),
			 name);
  else
    {
      if (negate)
	forward_file_checks &= ~val;
      else
	forward_file_checks |= val;
    }
  return 0;
}

static int
cb_forward_file_checks (mu_debug_t debug, void *data, mu_config_value_t *arg)
{
  return mu_cfg_string_value_cb (debug, arg, cb2_forward_file_checks, data);
}

static int
cb_script_language (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  script_handler = script_lang_handler (val->v.string);
  if (!script_handler)
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			   _("unsupported language: %s"),
			   val->v.string);
      return 1;
    }
  return 0;
}

static int
cb_script_pattern (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  
  switch (script_register (val->v.string))
    {
    case 0:
      break;

    case EINVAL:
      mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			   _("%s has unknown file suffix"),
			   val->v.string);
      break;

    default:
      mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			   _("error registering script"));
    }
  return 0;
}

struct mu_cfg_param filter_cfg_param[] = {
  { "language", mu_cfg_callback, NULL, 0, cb_script_language,
    N_("Set script language.") },
  { "pattern", mu_cfg_callback, NULL, 0, cb_script_pattern,
    N_("Set script pattern.") },
  { NULL }
};
    
struct mu_cfg_param maidag_cfg_param[] = {
  { "exit-multiple-delivery-success", mu_cfg_bool, &multiple_delivery, 0, NULL,
    N_("In case of multiple delivery, exit with code 0 if at least one "
       "delivery succeeded.") },
  { "exit-quota-tempfail", mu_cfg_bool, &ex_quota_tempfail, 0, NULL,
    N_("Indicate temporary failure if the recipient is over his mail quota.")
  },
#ifdef USE_DBM
  { "quota-db", mu_cfg_string, &quotadbname, 0, NULL,
    N_("Name of DBM quota database file."),
    N_("file") },
#endif
#ifdef USE_SQL
  { "quota-query", mu_cfg_string, &quota_query, 0, NULL,
    N_("SQL query to retrieve mailbox quota.  This is deprecated, use "
       "sql { ... } instead."),
    N_("query") },
#endif
  { "message-id-header", mu_cfg_string, &message_id_header, 0, NULL,
    N_("When logging Sieve actions, identify messages by the value of "
       "this header."),
    N_("name") },
  { "debug", mu_cfg_callback, NULL, 0, cb_debug,
    N_("Set maidag debug level.  Debug level consists of one or more "
       "of the following letters:\n"
       "  g - guimb stack traces\n"
       "  t - sieve trace (MU_SIEVE_DEBUG_TRACE)\n"
       "  i - sieve instructions trace (MU_SIEVE_DEBUG_INSTR)\n"
       "  l - sieve action logs\n") },
  { "stderr", mu_cfg_bool, &log_to_stderr, 0, NULL,
    N_("Log to stderr instead of syslog.") },
  { "forward-file", mu_cfg_string, &forward_file, 0, NULL,
    N_("Process forward file.") },
  { "forward-file-checks", mu_cfg_callback, NULL, 0, cb_forward_file_checks,
    N_("Configure safety checks for the forward file."),
    N_("arg: list") },
/* LMTP support */
  { "lmtp", mu_cfg_bool, &lmtp_mode, 0, NULL,
    N_("Run in LMTP mode.") },
  { "group", mu_cfg_callback, &lmtp_groups, 0, cb_group,
    N_("In LMTP mode, retain these supplementary groups."),
    N_("groups: list of string") },
  { "listen", mu_cfg_string, &lmtp_url_string, 0, NULL,
    N_("In LMTP mode, listen on the given URL.  Valid URLs are:\n"
       "   tcp://<address: string>:<port: number> (note that port is "
       "mandatory)\n"
       "   file://<socket-file-name>\n"
       "or socket://<socket-file-name>"),
    N_("url") },
  { "reuse-address", mu_cfg_bool, &reuse_lmtp_address, 0, NULL,
    N_("Reuse existing address (LMTP mode).  Default is \"yes\".") },
  { "filter", mu_cfg_section, NULL, 0, NULL,
    N_("Add a message filter") },
  { ".server", mu_cfg_section, NULL, 0, NULL,
    N_("LMTP server configuration.") },
  TCP_WRAPPERS_CONFIG
  { NULL }
};

static void
maidag_cfg_init ()
{
  struct mu_cfg_section *section;
  if (mu_create_canned_section ("filter", &section) == 0)
    {
      section->docstring = N_("Add new message filter.");
      mu_cfg_section_add_params (section, filter_cfg_param);
    }
}

/* FIXME: These are for compatibility with MU 2.0.
   Remove in 2.2 */
extern mu_record_t mu_remote_smtp_record;
extern mu_record_t mu_remote_sendmail_record;
extern mu_record_t mu_remote_prog_record;


int
main (int argc, char *argv[])
{
  int arg_index;
  mu_debug_t debug;

  /* Preparative work: close inherited fds, force a reasonable umask
     and prepare a logging. */
  close_fds ();
  umask (0077);

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  /* Default locker settings */
  mu_locker_set_default_flags (MU_LOCKER_PID|MU_LOCKER_RETRY,
			    mu_locker_assign);
  mu_locker_set_default_retry_timeout (1);
  mu_locker_set_default_retry_count (300);

  /* Register needed modules */
  MU_AUTH_REGISTER_ALL_MODULES ();

  /* Register all supported mailbox and mailer formats */
  mu_register_all_formats ();
  mu_registrar_record (mu_smtp_record);

  /* FIXME: These are for compatibility with MU 2.0.
     Remove in 2.1 */
  mu_registrar_record (mu_remote_smtp_record);
  mu_registrar_record (mu_remote_sendmail_record);
  mu_registrar_record (mu_remote_prog_record);
  
  mu_gocs_register ("sieve", mu_sieve_module_init);

  mu_tcpwrapper_cfg_init ();
  mu_acl_cfg_init ();
  mu_m_server_cfg_init ();
  maidag_cfg_init ();
  
  /* Parse command line */
  mu_argp_init (program_version, NULL);

  mu_m_server_create (&server, program_version);
  mu_m_server_set_conn (server, lmtp_connection);
  mu_m_server_set_prefork (server, mu_tcp_wrapper_prefork);
  mu_m_server_set_mode (server, MODE_INTERACTIVE);
  mu_m_server_set_max_children (server, 20);
  mu_m_server_set_timeout (server, 600);
  
  if (mu_app_init (&argp, maidag_argp_capa, maidag_cfg_param, 
		   argc, argv, 0, &arg_index, server))
    exit (EX_CONFIG);

  current_uid = getuid ();

  if (log_to_stderr == -1)
    log_to_stderr = url_option || (!lmtp_mode && (current_uid != 0));
  
  mu_diag_get_debug (&debug);
  if (!log_to_stderr)
    {
      openlog (MU_LOG_TAG (), LOG_PID, mu_log_facility);
      mu_debug_set_print (debug, mu_diag_syslog_printer, NULL);
      mu_debug_default_printer = mu_debug_syslog_printer;
    }
  else
    {
      mu_debug_set_print (debug, mu_diag_stderr_printer, NULL);
      mu_debug_default_printer = mu_debug_stderr_printer;
    }

  argc -= arg_index;
  argv += arg_index;

  if (lmtp_mode && !url_option)
    {
      if (argc)
	{
	  mu_error (_("too many arguments"));
	  return EX_USAGE;
	}
      return maidag_lmtp_server ();
    }
  else 
    {
      if (current_uid)
	{
	  if (url_option)
	    {
	      /* FIXME: Verify if the urls are deliverable? */
	    }
	  else
	    {
	      static char *s_argv[2];
	      struct mu_auth_data *auth = mu_get_auth_by_uid (current_uid);
	      
	      if (!current_uid)
		{
		  mu_error (_("cannot get username"));
		  return EX_UNAVAILABLE;
		}
	      
	      if (argc > 0 && strcmp (auth->name, argv[0]))
		{
		  mu_error (_("recipients given when running as non-root"));
		  return EX_USAGE;
		}
	      s_argv[0] = auth->name;
	      argv = s_argv;
	      argc = 1;
	    }
	}
      return maidag_stdio_delivery (argc, argv);
    }
}
  

