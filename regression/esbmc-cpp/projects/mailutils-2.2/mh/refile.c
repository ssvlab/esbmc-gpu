/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010
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

/* MH refile command */

#include <mh.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

const char *program_version = "refile (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH refile")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("messages folder [folder...]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"draft",   ARG_DRAFT, NULL, 0,
   N_("use <mh-dir>/draft as the source message")},
  {"copy",    ARG_LINK, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("preserve the source folder copy")},
  {"link",    0, NULL, OPTION_ALIAS, NULL},
  {"preserve", ARG_PRESERVE, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* try to preserve message sequence numbers")},
  {"source", ARG_SOURCE, N_("FOLDER"), 0,
   N_("specify source folder; it will become the current folder after the program exits")},
  {"src", 0, NULL, OPTION_ALIAS, NULL},
  {"file", ARG_FILE, N_("FILE"), 0, N_("use FILE as the source message")},
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"file",     2, 0, "input-file"},
  {"draft",    1, 0, NULL },
  {"link",     1, MH_OPT_BOOL, NULL },
  {"preserve", 1, MH_OPT_BOOL, NULL },
  {"src",      1, 0, "+folder" },
  { 0 }
};

int link_flag = 0;
int preserve_flag = 0;
char *source_file = NULL;
mu_list_t folder_name_list = NULL;
mu_list_t folder_mbox_list = NULL;

void
add_folder (const char *folder)
{
  if (!folder_name_list && mu_list_create (&folder_name_list))
    {
      mu_error (_("cannot create folder list"));
      exit (1);
    }
  mu_list_append (folder_name_list, strdup (folder));
}

void
open_folders ()
{
  int rc;
  mu_iterator_t itr;

  if (!folder_name_list)
    {
      mu_error (_("no folder specified"));
      exit (1);
    }

  if ((rc = mu_list_create (&folder_mbox_list)) != 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_list_create", NULL, rc);
      exit (1);
    }

  if ((rc = mu_list_get_iterator (folder_name_list, &itr)) != 0)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_list_get_iterator", NULL, rc);
      exit (1);
    }

  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      char *name = NULL;
      mu_mailbox_t mbox;
      
      mu_iterator_current (itr, (void **)&name);
      mbox = mh_open_folder (name, 1);
      mu_list_append (folder_mbox_list, mbox);
      free (name);
    }
  mu_iterator_destroy (&itr);
  mu_list_destroy (&folder_name_list);
}

void
enumerate_folders (void (*f) (void *, mu_mailbox_t), void *data)
{
  mu_iterator_t itr;

  if (mu_list_get_iterator (folder_mbox_list, &itr))
    {
      mu_error (_("cannot create iterator"));
      exit (1);
    }

  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      mu_mailbox_t mbox;
      mu_iterator_current (itr, (void **)&mbox);
      (*f) (data, mbox);
    }
  mu_iterator_destroy (&itr);
}
  
void
_close_folder (void *unused, mu_mailbox_t mbox)
{
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
}

void
close_folders ()
{
  enumerate_folders (_close_folder, NULL);
}

static int
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      add_folder (arg);
      break;

    case ARG_DRAFT:
      source_file = "draft";
      break;

    case ARG_LINK:
      link_flag = is_true(arg);
      break;
      
    case ARG_PRESERVE:
      mh_opt_notimpl_warning ("-preserve");
      preserve_flag = is_true(arg);
      break;
	
    case ARG_SOURCE:
      mh_set_current_folder (arg);
      break;
      
    case ARG_FILE:
      source_file = arg;
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

void
refile_folder (void *data, mu_mailbox_t mbox)
{
  mu_message_t msg = data;
  int rc;
  
  rc = mu_mailbox_append_message (mbox, msg);
  if (rc)
    {
      mu_error (_("error appending message: %s"), mu_strerror (rc));
      exit (1);
    }
}

void
refile (mu_message_t msg)
{
  enumerate_folders (refile_folder, msg);
}

void
refile_iterator (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  enumerate_folders (refile_folder, msg);
  if (!link_flag)
    {
      mu_attribute_t attr;
      mu_message_get_attribute (msg, &attr);
      mu_attribute_set_deleted (attr);
    }
}

int
main (int argc, char **argv)
{
  int index;
  mh_msgset_t msgset;
  mu_mailbox_t mbox;
  int status, i, j;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  /* Collect any surplus folders */
  for (i = j = 0; i < argc; i++)
    {
      if (argv[i][0] == '+')
	add_folder (argv[i]);
      else
	argv[j++] = argv[i];
    }
  argv[j] = NULL;
  argc = j;
  
  open_folders ();

  if (source_file)
    {
      mu_message_t msg;
      
      if (argc > 0)
	{
	  mu_error (_("both message set and source file given"));
	  exit (1);
	}
      msg = mh_file_to_message (mu_folder_directory (), source_file);
      refile (msg);
      if (!link_flag)
	unlink (source_file);
      status = 0;
    }
  else
    {
      mbox = mh_open_folder (mh_current_folder (), 0);
      mh_msgset_parse (mbox, &msgset, argc, argv, "cur");

      status = mh_iterate (mbox, &msgset, refile_iterator, NULL);
 
      mu_mailbox_expunge (mbox);
      mu_mailbox_close (mbox);
      mu_mailbox_destroy (&mbox);
    }

  close_folders ();
  
  return status;
}
