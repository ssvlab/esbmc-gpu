/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2008, 2009, 2010 Free Software Foundation,
   Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <mimeview.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "mailutils/libargp.h"
#include "mailutils/argcv.h"

#include "mailcap.h"

const char *program_version = "mimeview (" PACKAGE_STRING ")";
static char doc[] = N_("GNU mimeview -- display files, using mailcap mechanism.")
"\v"     
N_("Default mime.types file is ") DEFAULT_CUPS_CONFDIR "/mime.types"
N_("\n\nDebug flags are:\n\
  g - Mime.types parser traces\n\
  l - Mime.types lexical analyzer traces\n\
  0-9 - Set debugging level\n");

#define OPT_METAMAIL 256

static struct argp_option options[] = {
  {"no-ask", 'a', N_("TYPE-LIST"), OPTION_ARG_OPTIONAL,
   N_("do not ask for confirmation before displaying files, or, if TYPE-LIST is given, do not ask for confirmation before displaying such files whose MIME type matches one of the patterns from TYPE-LIST"), 0},
  {"no-interactive", 'h', NULL, 0,
   N_("disable interactive mode"), 0 },
  {"print", 0, NULL, OPTION_ALIAS, NULL, 0 },
  {"debug",  'd', N_("FLAGS"),  OPTION_ARG_OPTIONAL,
   N_("enable debugging output"), 0},
  {"mimetypes", 't', N_("FILE"), 0,
   N_("use this mime.types file"), 0},
  {"dry-run", 'n', NULL, 0,
   N_("do not do anything, just print what whould be done"), 0},
  {"metamail", OPT_METAMAIL, N_("FILE"), OPTION_ARG_OPTIONAL,
   N_("use metamail to display files"), 0},
  {0, 0, 0, 0}
};

int debug_level;       /* Debugging level set by --debug option */
static int dry_run;    /* Dry run mode */
static char *metamail; /* Name of metamail program, if requested */
static char *mimetypes_config = DEFAULT_CUPS_CONFDIR;
static char *no_ask_types;  /* List of MIME types for which no questions
			       should be asked */
static int interactive = -1; 
char *mimeview_file;       /* Name of the file to view */
FILE *mimeview_fp;     /* Its descriptor */

static void
set_debug_flags (mu_debug_t debug, const char *arg)
{
  for (; *arg; arg++)
    {
      switch (*arg)
	{
	case 'l':
	  mimetypes_lex_debug (1);
	  break;

	case 'g':
	  mimetypes_gram_debug (1);
	  break;
	  
	default:
	  debug_level = *arg - '0';
	}
    }
}  

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  static mu_list_t lst;

  switch (key)
    {
    case ARGP_KEY_INIT:
      mimetypes_lex_debug (0);
      mimetypes_gram_debug (0);
      if (interactive == -1)
	interactive = isatty (fileno (stdin));
      mu_argp_node_list_init (&lst);
      break;

    case ARGP_KEY_FINI:
      if (dry_run && !debug_level)
	debug_level = 1;
      mu_argp_node_list_finish (lst, NULL, NULL);
      break;

    case 'a':
      no_ask_types = arg ? arg : "*";
      setenv ("MM_NOASK", arg, 1); /* In case we are given --metamail option */
      break;
      
    case 'd':
      mu_argp_node_list_new (lst, "debug", arg ? arg : "9");
      break;

    case 'h':
      interactive = 0;
      break;
      
    case 'n':
      dry_run = 1;
      break;
      
    case 't':
      mu_argp_node_list_new (lst, "mimetypes", arg);
      break;

    case OPT_METAMAIL:
      mu_argp_node_list_new (lst, "metamail", arg ? arg : "metamail");
      break;
      
    default: 
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {
  options,
  parse_opt,
  N_("FILE [FILE ...]"),
  doc,
  NULL,
  NULL, NULL
};


static int
cb_debug (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  set_debug_flags (debug, val->v.string);
  return 0;
}

struct mu_cfg_param mimeview_cfg_param[] = {
  { "debug", mu_cfg_callback, NULL, 0, cb_debug,
    N_("Set debug verbosity level."),
    N_("flags") },
  { "mimetypes", mu_cfg_string, &mimetypes_config, 0, NULL,
    N_("Use this mime.types file."),
    N_("file") },
  { "metamail", mu_cfg_string, &metamail, 0, NULL,
    N_("Use this program to display files."),
    N_("prog") },
  { NULL }
};



static const char *capa[] = {
  "common",
  "debug",
  "license",
  NULL
};

static int
open_file (char *name)
{
  struct stat st;
  if (stat (name, &st))
    {
      mu_error (_("cannot stat `%s': %s"), name, mu_strerror (errno));
      return -1;
    }
  if (!S_ISREG (st.st_mode) && !S_ISLNK (st.st_mode))
    {
      mu_error (_("not a regular file or symbolic link: `%s'"), name);
      return -1;
    }

  mimeview_file = name;
  mimeview_fp = fopen (name, "r");
  if (mimeview_fp == NULL)
    {
      mu_error (_("Cannot open `%s': %s"), name, mu_strerror (errno));
      return -1;
    }
  return 0;
}

void
close_file ()
{
  fclose (mimeview_fp);
}

void
display_file (const char *type)
{
  int status;
  
  if (metamail)
    {
      char *argv[7];
      
      argv[0] = "metamail";
      argv[1] = "-b";

      argv[2] = interactive ? "-p" : "-h";
      
      argv[3] = "-c";
      argv[4] = (char*) type;
      argv[5] = mimeview_file;
      argv[6] = NULL;
      
      if (debug_level)
	{
	  char *string;
	  mu_argcv_string (6, argv, &string);
	  printf (_("Executing %s...\n"), string);
	  free (string);
	}
      
      if (!dry_run)
	mu_spawnvp (metamail, argv, &status);
    }
  else
    {
      mu_stream_t stream;
      mu_header_t hdr;
      char *text;

      asprintf (&text, "Content-Type: %s\n", type);
      status = mu_header_create (&hdr, text, strlen (text), NULL);
      if (status)
	mu_error (_("cannot create header: %s"), mu_strerror (status));
      else
	{
	  mu_stdio_stream_create (&stream, mimeview_fp,
			       MU_STREAM_READ|MU_STREAM_SEEKABLE|MU_STREAM_NO_CLOSE);
	  mu_stream_open (stream);
	  
	  display_stream_mailcap (mimeview_file, stream, hdr,
				  no_ask_types, interactive, dry_run,
				  debug_level);
	  
	  mu_stream_close (stream);
	  mu_stream_destroy (&stream, mu_stream_get_owner (stream));

	  mu_header_destroy (&hdr, mu_header_get_owner (hdr));
	}
    }
}

int
main (int argc, char **argv)
{
  int index;
  
  MU_APP_INIT_NLS ();
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, capa, mimeview_cfg_param, 
		   argc, argv, 0, &index, NULL))
    exit (1);

  argc -= index;
  argv += index;

  if (argc == 0)
    {
      mu_error (_("no files given"));
      return 1;
    }

  if (mimetypes_parse (mimetypes_config))
    return 1;
  
  while (argc--)
    {
      const char *type;
      
      if (open_file (*argv++))
	continue;
      type = get_file_type ();
      DEBUG (1, ("%s: %s\n", mimeview_file, type ? type : "?"));
      if (type)
	display_file (type);
      close_file ();
    }
  
  return 0;
}
