/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2008, 2009, 2010 Free Software Foundation,
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

#include <mh.h>

const char *program_version = "whom (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH whom")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[file]";

/* GNU options */
static struct argp_option options[] = {
  {"alias",         ARG_ALIAS,         N_("FILE"), 0,
   N_("specify additional alias file") },
  {"draft",         ARG_DRAFT,         NULL, 0,
   N_("use prepared draft") },
  {"draftfolder",   ARG_DRAFTFOLDER,   N_("FOLDER"), 0,
   N_("specify the folder for message drafts") },
  {"draftmessage",  ARG_DRAFTMESSAGE,  NULL, 0,
   N_("treat the arguments as a list of messages from the draftfolder") },
  {"nodraftfolder", ARG_NODRAFTFOLDER, NULL, 0,
   N_("undo the effect of the last --draftfolder option") },
  {"check",         ARG_CHECK,         N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("check if addresses are deliverable") },
  {"nocheck",       ARG_NOCHECK,       NULL, OPTION_HIDDEN, "" },

  {"license",       ARG_LICENSE,       0,    0,
   N_("display software license"), -1},

  {NULL}
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"alias",         1, 0, "aliasfile" },
  {"draft",         5, 0, NULL },
  {"draftfolder",   6, 0, "folder" },
  {"draftmessage",  6, 0, "message"},
  {"nodraftfolder", 3, 0, NULL },
  {"check",         1, MH_OPT_BOOL, NULL},
  {NULL}
};

static int check_recipients;
static int use_draft;            /* Use the prepared draft */
static const char *draft_folder; /* Use this draft folder */

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_ALIAS:
      mh_alias_read (arg, 1);
      break;
      
    case ARG_DRAFT:
      use_draft = 1;
      break;
	
    case ARG_DRAFTFOLDER:
      draft_folder = arg;
      break;
      
    case ARG_NODRAFTFOLDER:
      draft_folder = NULL;
      break;
      
    case ARG_DRAFTMESSAGE:
      if (!draft_folder)
	draft_folder = mh_global_profile_get ("Draft-Folder",
					      mu_folder_directory ());
      break;

    case ARG_CHECK:
      check_recipients = is_true (arg);
      break;

    case ARG_NOCHECK:
      check_recipients = 0;
      break;

    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

int
main (int argc, char **argv)
{
  int index;
  char *name = "draft";
  
  MU_APP_INIT_NLS ();
  
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  if (!use_draft && argc > 0)
    name = argv[0];

  if (!draft_folder)
    draft_folder = mh_global_profile_get ("Draft-Folder",
					  mu_folder_directory ());

  
  return mh_whom (mh_expand_name (draft_folder, name, 0), check_recipients) ?
          1 : 0;
}
