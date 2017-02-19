/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2005, 2007, 2008, 2009, 2010 Free Software
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

/* MH mhpath command */

#include <mh.h>

const char *program_version = "mhpath (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH mhpath")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[+folder] [msgs]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  { 0 }
};

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER:
      mh_set_current_folder (arg);
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
mhpath (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  size_t uid;
      
  mh_message_number (msg, &uid);
  printf ("%s/%s\n", (char*) data, mu_umaxtostr (0, uid));
}

int
main (int argc, char **argv)
{
  int index = 0;
  mu_mailbox_t mbox = NULL;
  mu_url_t url = NULL;
  char *mhdir;
  size_t total;
  mh_msgset_t msgset;
  int status;
  const char *current_folder;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  current_folder = mh_current_folder ();
  /* If  the  only argument  is `+', your MH Path is output; this
     can be useful is shell scripts. */
  if (current_folder[0] == 0)
    {
      printf ("%s\n", mu_folder_directory ());
      exit (0);
    }
  
  mbox = mh_open_folder (current_folder, 0);

  mu_mailbox_messages_count (mbox, &total);
  mu_mailbox_get_url (mbox, &url);
  mhdir = (char*) mu_url_to_string (url);
  if (strncmp (mhdir, "mh:", 3) == 0)
    mhdir += 3;

  /* If no `msgs' are specified, mhpath outputs the folder pathname
     instead.  */
  if (index == argc)
    {
      printf ("%s\n", mhdir);
      exit (0);
    }
  
  /* the name "new" has  been  added  to  mhpath's  list  of
     reserved  message  names  (the others are "first", "last",
     "prev", "next", "cur", and "all").   The  new  message  is
     equivalent  to  the  message  after  the last message in a
     folder (and equivalent to 1 in a folder without messages).
     The  "new"  message  may  not be used as part of a message
     range. */
  
  if (argc - index == 1 && strcmp (argv[index], "new") == 0)
    {
      mu_message_t msg = NULL;
      size_t num;
      
      mu_mailbox_get_message (mbox, total, &msg);
      mh_message_number (msg, &num);
      printf ("%s/%s\n", mhdir, mu_umaxtostr (0, num + 1));
      exit (0);
    }
      
  /* Mhpath  expands  and  sorts  the  message  list `msgs' and
     writes the full pathnames of the messages to the  standard
     output separated by newlines. */
  mh_msgset_parse (mbox, &msgset, argc - index, argv + index, "cur");
  status = mh_iterate (mbox, &msgset, mhpath, mhdir);
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return status;
}
