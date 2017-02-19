/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2008, 2009, 2010 Free Software Foundation, Inc.

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

/* MH whatnow command */

#include <mh.h>

const char *program_version = "whatnow (" PACKAGE_STRING ")";
static char doc[] = "GNU MH whatnow";
static char args_doc[] = N_("[FILE]");

/* GNU options */
static struct argp_option options[] = {
  {"draftfolder", ARG_DRAFTFOLDER, N_("FOLDER"), 0,
   N_("specify the folder for message drafts")},
  {"nodraftfolder", ARG_NODRAFTFOLDER, 0, 0,
   N_("undo the effect of the last --draftfolder option")},
  {"draftmessage" , ARG_DRAFTMESSAGE, N_("MSG"), 0,
   N_("invoke the draftmessage facility")},
  {"editor",  ARG_EDITOR, N_("PROG"), 0, N_("set the editor program to use")},
  {"noedit", ARG_NOEDIT, 0, 0, N_("suppress the initial edit")},
  {"prompt", ARG_PROMPT, N_("STRING"), 0, N_("set the prompt")},

  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},

  { NULL }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"draftfolder", 6, MH_OPT_ARG, "folder"},
  {"nodraftfolder", 3, },
  {"draftmessage", 6, },
  {"editor", 1, MH_OPT_ARG, "program"},
  {"noedit", 3, },
  {"prompt", 1 },
  { 0 }
};

struct mh_whatnow_env wh_env = { 0 };
static int initial_edit = 1;
static char *draftmessage = "cur";

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARGP_KEY_INIT:
      wh_env.draftfolder = mh_global_profile_get ("Draft-Folder",
						  mu_folder_directory ());
      break;
      
    case ARG_DRAFTFOLDER:
      wh_env.draftfolder = arg;
      break;
      
    case ARG_EDITOR:
      wh_env.editor = arg;
      break;
      
    case ARG_NODRAFTFOLDER:
      wh_env.draftfolder = NULL;
      break;

    case ARG_NOEDIT:
      initial_edit = 0;
      break;

    case ARG_DRAFTMESSAGE:
      draftmessage = arg;
      break;

    case ARG_PROMPT:
      wh_env.prompt = arg;
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
  
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);
  argc -= index;
  argv += index;
  if (argc)
    wh_env.draftfile = argv[0];
  else if (wh_env.draftfolder)
    {
      if (mh_draft_message (wh_env.draftfolder, draftmessage,
			    &wh_env.file))
	return 1;
    }
  else
    wh_env.draftfile = mh_expand_name (wh_env.draftfolder, "draft", 0);
  wh_env.draftfile = wh_env.file;
  wh_env.msg = getenv ("mhaltmsg");
  return mh_whatnow (&wh_env, initial_edit);
}
  
