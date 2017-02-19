/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2008, 2009, 2010 Free Software
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

/* MH annotate command */

#include <mh.h>

const char *program_version = "anno (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH anno")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[msg [msg...]]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"inplace", ARG_INPLACE, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* annotate the message in place")},
  {"noinplace", ARG_NOINPLACE, NULL, OPTION_HIDDEN,  "" },
  {"date", ARG_DATE, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("add FIELD: date header") },
  {"nodate", ARG_NODATE, NULL, OPTION_HIDDEN, "" },
  {"component", ARG_COMPONENT, N_("FIELD"), 0,
   N_("add this FIELD to the message header") },
  {"text", ARG_TEXT, N_("STRING"), 0,
   N_("field value for the component") },
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { NULL }
};

struct mh_option mh_option[] = {
  {"inplace",   1, MH_OPT_BOOL },
  {"date",      1, MH_OPT_BOOL },
  {"component", 1, MH_OPT_ARG, "field"},
  {"text",      1, MH_OPT_ARG, "body"},
  { NULL }
};

static int inplace;       /* Annotate the message in place */
static int anno_date = 1; /* Add date to the annotation */
static char *component;   /* header field */
static char *anno_text;   /* header field value */

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_INPLACE:
      mh_opt_notimpl_warning ("-inplace");
      inplace = is_true (arg);
      break;

    case ARG_NOINPLACE:
      mh_opt_notimpl_warning ("-noinplace");
      inplace = 0;
      break;

    case ARG_DATE:
      anno_date = is_true (arg);
      break;

    case ARG_NODATE:
      anno_date = 0;
      break;

    case ARG_COMPONENT:
      component = arg;
      break;

    case ARG_TEXT:
      mh_quote (arg, &anno_text);
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
anno (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  mh_annotate (msg, component, anno_text, anno_date);
}

int
main (int argc, char **argv)
{
  int rc;
  int index;
  mu_mailbox_t mbox;
  mh_msgset_t msgset;
  size_t len;
  
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  mbox = mh_open_folder (mh_current_folder (), 0);

  if (!component)
    {
      size_t n;
      
      printf (_("Component name: "));
      if (getline (&component, &n, stdin) <= 0 || *component == 0)
	exit (1);
    }

  if (!anno_text && !anno_date)
    exit (0);

  len = strlen (component);
  if (len > 0 && component[len-1] == ':')
    component[len-1] = 0;
  
  argc -= index;
  argv += index;
  
  mh_msgset_parse (mbox, &msgset, argc, argv, "cur");
  rc = mh_iterate (mbox, &msgset, anno, NULL);

  mh_msgset_current (mbox, &msgset, 0);
  mh_global_save_state ();
  mu_mailbox_sync (mbox);
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return rc;
}
      

				  
  



