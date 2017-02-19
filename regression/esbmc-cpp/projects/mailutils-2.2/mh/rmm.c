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

/* MH rmm command */

#include <mh.h>

const char *program_version = "rmm (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH rmm")"\v"
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
rmm (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  mu_attribute_t attr;
  mu_message_get_attribute (msg, &attr);
  mu_attribute_set_deleted (attr);
}

int
main (int argc, char **argv)
{
  int index = 0;
  mu_mailbox_t mbox;
  mh_msgset_t msgset;
  int status;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  mbox = mh_open_folder (mh_current_folder (), 0);

  mh_msgset_parse (mbox, &msgset, argc - index, argv + index, "cur");

  status = mh_iterate (mbox, &msgset, rmm, NULL);

  mu_mailbox_expunge (mbox);
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return status;
}

