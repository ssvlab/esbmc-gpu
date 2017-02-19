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

#include "guimb.h"
#include "mailutils/libargp.h"

char *program_file;
char *program_expr;
int debug_guile;
char *user_name;
char *default_mailbox;

char * who_am_i ();

static int g_size;
static int g_argc;
static char **g_argv;

#define ARG_INC 16

void
append_arg (char *arg)
{
  if (g_argc == g_size)
    {
      g_size += ARG_INC;
      g_argv = realloc (g_argv, g_size * sizeof (g_argv[0]));
      if (!g_argv)
	{
	  util_error (_("not enough memory"));
	  exit (1);
	}
    }
  g_argv[g_argc++] = arg;
}

static struct argp_option options[] = {
  {NULL, 0, NULL, 0,
   /* TRANSLATORS: (command-line) is the name of Guile function. Do not
      translate it.
   */
   N_("The following switches stop argument processing, and pass all\n"
   "remaining arguments as the value of (command-line):"), 1},
  {"code", 'c', N_("EXPR"), 0, N_("execute given scheme expression"), 1},
  {"source", 's', N_("PROGFILE"), 0,
   N_("load Scheme source code from PROGFILE and exit"), 1},
  {NULL, 0, NULL, 0,
   N_("The following options do not change the order of options parsing:"), 2},
  {"expression", 'e', N_("EXPR"), 0, N_("execute given scheme expression"), 2},
  {"file", 'f', N_("PROGFILE"), 0,
   N_("load Scheme source code from PROGFILE and exit"), 2},
  {NULL, 0, NULL, 0, N_("Other options:"), 3},
  {"debug", 'd', NULL, 0, N_("start with debugging evaluator and backtraces"), 3},
  {"guile-arg", 'g', N_("ARG"), 0,
   N_("append ARG to the command line passed to Guile"), 3},
  {"mailbox", 'M', N_("NAME"), 0, N_("set default mailbox name"), 3},
  {"user", 'u', N_("NAME"), OPTION_ARG_OPTIONAL,
   N_("act as local MDA for user NAME"), 3},
  {0, 0, 0, 0}
};

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'c':
      program_expr = arg;
      *(int *)state->input = state->next;
      state->next = state->argc;
      break;
      
    case 's':
      program_file = arg;
      *(int *)state->input = state->next;
      state->next = state->argc;
      break;

    case 'f':
      program_file = arg;
      break;

    case 'e':
      program_expr = arg;
      break;

    case 'd':
      debug_guile = 1;
      break;

    case 'g':
      append_arg (arg);
      break;

    case 'M':
      default_mailbox = arg;
      break;

    case 'u':
      user_name = arg ? arg : who_am_i ();
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

const char *program_version = "guimb (" PACKAGE_STRING ")";
static char doc[] =
N_("GNU guimb -- process contents of the specified mailboxes "
"using a Scheme program or expression.");
static char args_doc[] = N_("[mailbox...]");

static struct argp argp = {
  options,
  parse_opt,
  args_doc,
  doc,
  NULL,
  NULL, NULL
};

static const char *guimb_argp_capa[] = {
  "common",
  "debug",
  "mailbox",
  "locking",
  "license",
  NULL
};

char *main_sym = "mailutils-main";

int
main (int argc, char *argv[])
{
  int rc;
  int c = argc;
  int index;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  /* Register the desired formats. */
  mu_register_all_formats ();

  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, guimb_argp_capa, NULL, argc, argv, 0, &index, &c))
    exit (1);

  for (; c < argc; c++)
    append_arg (argv[c]);

  if (!user_name)
    user_name = who_am_i ();
  
  if (!program_file && !program_expr)
    {
      mu_error (_("At least one of -fecs must be used. Try guimb --help for more info."));
      exit (1);
    }
    
  if (!argv[index])
    {
      if (default_mailbox)
	append_arg (default_mailbox);
      collect_open_default ();
    }
  else
    {
      collect_open_mailbox_file ();

      if (argv[index])
	{
	  for (; argv[index]; index++)
	    {
	      append_arg (argv[index]);
	      collect_append_file (argv[index]);
	    }
	}
      else 
	collect_append_file ("-");
    }

  /* Finish creating input mailbox */
  collect_create_mailbox ();

  mu_guile_init (debug_guile);
  if (program_file)
    mu_guile_load (program_file, g_argc, g_argv);
  if (program_expr)
    mu_guile_eval (program_expr);

  rc = mu_guile_mailbox_apply (mbox, main_sym);
  switch (rc)
    {
    case 0:
      collect_output ();
      break;

    case MU_ERR_NOENT:
      mu_error (_("%s not defined"), main_sym);
      break;
      
    case EINVAL:
      mu_error (_("%s is not a procedure object"), main_sym);
      break;

    case MU_ERR_FAILURE:
      mu_error (_("execution of %s failed"), main_sym);
      break;
      
    default:
      mu_error (_("unrecognized error"));
      break;
    }

  collect_drop_mailbox ();
  
  return !!rc;
}

char *
who_am_i ()
{
  char *name = getenv ("LOGNAME");
  if (!name)
    {
      name = getenv ("USER");
      if (!name)
	name = strdup (getlogin ());
    }
  return name;
}

