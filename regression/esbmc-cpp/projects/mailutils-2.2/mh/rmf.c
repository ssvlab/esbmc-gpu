/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2005, 2006, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

/* MH rmf command */

#include <mh.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <dirent.h>

const char *program_version = "rmf (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH rmf")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[+folder]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify the folder to delete")},
  {"interactive", ARG_INTERACTIVE, N_("BOOL"), OPTION_ARG_OPTIONAL,
    N_("interactive mode: ask for confirmation before removing each folder")},
  {"nointeractive", ARG_NOINTERACTIVE, NULL, OPTION_HIDDEN, ""},
  {"recursive", ARG_RECURSIVE, NULL, 0,
   N_("recursively delete all subfolders")},
  {"norecursive", ARG_NORECURSIVE, NULL, OPTION_HIDDEN, ""},
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"interactive", 1, MH_OPT_BOOL, NULL},
  { 0 }
};

int explicit_folder; /* Was the folder explicitly given */
int interactive; /* Ask for confirmation before deleting */
int recurse;     /* Recursively process all the sub-directories */

static char *cur_folder_path; /* Full pathname of the current folder */
static char *folder_name;     /* Name of the (topmost) folder to be
				 deleted */

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER:
      explicit_folder = 1;
      folder_name = arg;
      break;

    case ARG_INTERACTIVE:
      interactive = is_true (arg);
      break;

    case ARG_NOINTERACTIVE:
      interactive = 0;
      break;
	
    case ARG_RECURSIVE:
      recurse = is_true (arg);
      break;
      
    case ARG_NORECURSIVE:
      recurse = 0;
      break;

    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static char *
current_folder_path ()
{
  mu_mailbox_t mbox = mh_open_folder (mh_current_folder (), 0);
  mu_url_t url;
  char *p;
  mu_mailbox_get_url (mbox, &url);
  p = (char*) mu_url_to_string (url);
  if (strncmp (p, "mh:", 3) == 0)
    p += 3;
  return p;
}

static int
rmf (const char *name)
{
  DIR *dir;
  struct dirent *entry;
  int failures = 0;
  
  dir = opendir (name);

  if (!dir)
    {
      mu_error (_("cannot scan folder %s: %s"), name, strerror (errno));
      return 1;
    }

  if (interactive && !mh_getyn (_("Remove folder %s"), name))
    exit (0);

  while ((entry = readdir (dir)))
    {
      char *p;
      struct stat st;

      if (strcmp (entry->d_name, ".") == 0
	  || strcmp (entry->d_name, "..") == 0)
	continue;
      
      asprintf (&p, "%s/%s", name, entry->d_name);
      if (stat (p, &st) < 0)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "stat", p, errno);
	}
      else if (S_ISDIR (st.st_mode))
	{
	  if (recurse)
	    failures += rmf (p);
	  else
	    {
	      printf ("%s: file `%s' not deleted, continuing...\n",
		      mu_program_name, p);
	      failures++;
	    }
	}
      else
	{
	  if (unlink (p))
	    {
	      mu_diag_funcall (MU_DIAG_ERROR, "unlink", p, errno);
	      failures++;
	    }
	}
      free (p);
    }
  closedir (dir);

  if (failures == 0)
    failures += rmdir (name);
  else
    printf ("%s: folder `%s' not removed\n",
	    mu_program_name, name);

  if (failures == 0)
    {
      if (cur_folder_path && strcmp (name, cur_folder_path) == 0)
	{
	  mh_set_current_folder ("inbox");
	  mh_global_sequences_drop ();
	  mh_global_save_state ();
	  printf ("[+inbox now current]\n");
	}
    }
  return failures;
}

int
main (int argc, char **argv)
{
  char *name;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, NULL);

  cur_folder_path = current_folder_path ();

  if (!explicit_folder)
    {
      interactive = 1;
      name = cur_folder_path;
    }
  else
    name = mh_expand_name (NULL, folder_name, 0);
  rmf (name);
  return 0;
}
