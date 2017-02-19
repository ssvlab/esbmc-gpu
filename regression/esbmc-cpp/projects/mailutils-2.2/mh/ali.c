/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Free
   Software Foundation, Inc.

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

/* MH ali command */

#include <mh.h>
#ifdef HAVE_TERMIOS_H
# include <termios.h>
#endif
#include <sys/ioctl.h>
#include <sys/stat.h>

const char *program_version = "ali (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH ali")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("aliases ...");

/* GNU options */
static struct argp_option options[] = {
  {"alias",  ARG_ALIAS, N_("FILE"), 0,
   N_("use the additional alias FILE")},
  {"noalias", ARG_NOALIAS, NULL, 0,
   N_("do not read the system alias file") },
  {"list", ARG_LIST, N_("BOOL"),  OPTION_ARG_OPTIONAL,
   N_("list each address on a separate line") },
  {"normalize", ARG_NORMALIZE, N_("BOOL"),  OPTION_ARG_OPTIONAL,
   N_("try to determine the official hostname for each address") },
  {"user", ARG_USER, N_("BOOL"),  OPTION_ARG_OPTIONAL,
   N_("list the aliases that expand to given addresses") },
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  { "alias", 1, MH_OPT_ARG, "aliasfile" },
  { "noalias", 3, },
  { "list", 1, MH_OPT_BOOL },
  { "normalize", 3, MH_OPT_BOOL },
  { "user", 1, MH_OPT_BOOL },
  { 0 }
};

static int list_mode;
static int user_mode;
static int normalize_mode;
static int nolist_mode;

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    case ARG_ALIAS:
      mh_alias_read (arg, 1);
      break;

    case ARG_NOALIAS:
      nolist_mode = 1;
      break;
      
    case ARG_LIST:
      list_mode = is_true (arg);
      break;
	
    case ARG_NORMALIZE:
      normalize_mode = is_true (arg);
      break;

    case ARG_USER:
      user_mode = is_true (arg);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static int
getcols ()
{
  struct winsize ws;

  ws.ws_col = ws.ws_row = 0;
  if ((ioctl(1, TIOCGWINSZ, (char *) &ws) < 0) || ws.ws_row == 0)
    {
      const char *columns = getenv ("COLUMNS");
      if (columns)
	ws.ws_col = strtol (columns, NULL, 10);
    }

  if (ws.ws_col == 0)
    ws.ws_col = 80;
  return ws.ws_col;
}

static void
ali_print_name_list (mu_list_t list, int off)
{
  mu_iterator_t itr;
  char *item;
  
  mu_list_get_iterator (list, &itr);
  
  if (list_mode)
    {
      mu_iterator_first (itr);
      mu_iterator_current (itr, (void **)&item);
      printf ("%s\n", item);
      for (mu_iterator_next (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
	{
	  int len;
	  mu_iterator_current (itr, (void **)&item);
	  len = off + strlen (item);
	  printf ("%*.*s\n", len, len, item);
	}
    }
  else
    {
      int ncol = getcols ();
      int n = off;
      
      mu_iterator_first (itr);

      for (;;)
	{
	  int len;

	  mu_iterator_current (itr, (void **)&item);
	  len = strlen (item) + 2;
	  if (n + len > ncol)
	    n = printf ("\n ");

	  len = printf ("%s", item);
	  mu_iterator_next (itr);
	  if (!mu_iterator_is_done (itr))
	    len += printf (", ");
	  else
	    break;
	  n += len;
	}
      printf ("\n");
    }
  mu_iterator_destroy (&itr);
}

static int
ali_print_alias (char *name, mu_list_t alias, void *data MU_ARG_UNUSED)
{
  int n;
  
  if (name)
    n = printf ("%s: ", name);
  else
    n = 0;

  ali_print_name_list (alias, n);
  return 0;
}

static void
ali_print_name (char *name)
{
  printf ("%s\n", name);
}

int
main (int argc, char **argv)
{
  int index;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  if (!nolist_mode)
    mh_read_aliases ();

  if (!user_mode)
    {
      if (argc == 0)
	mh_alias_enumerate (ali_print_alias, NULL);
      else
	{
	  int i;
	  for (i = 0; i < argc; i++)
	    {
	      mu_list_t al = NULL;
	      
	      if (mh_alias_get (argv[i], &al) == 0)
		{
		  ali_print_alias (NULL, al, NULL);
		  mu_list_destroy (&al);
		}
	      else
		ali_print_name (argv[i]);
	    }
	}
    }
  else
    {
      if (argc == 0)
	{
	  mu_error ("List of addresses is not given");
	  exit (1);
	}
      else
	{
	  int i;
	  for (i = 0; i < argc; i++)
	    {
	      mu_list_t nl = NULL;

	      if (mh_alias_get_alias (argv[i], &nl) == 0)
		{
		  ali_print_name_list (nl, 0);
		  mu_list_destroy (&nl);
		}
	      else
		printf ("%s\n", argv[i]);
	    }
	}
    }
  return 0;
}
