/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2006, 2007, 2008, 2009, 2010 Free Software
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

/* MH comp command */

#include <mh.h>
#include <sys/types.h>
#include <sys/stat.h>

const char *program_version = "comp (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH comp")"\v"
N_("Options marked with `*' are not yet implemented.\n"
"Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[msg]";

/* GNU options */
static struct argp_option options[] = {
  {"build",         ARG_BUILD, 0, 0,
   N_("build the draft and quit immediately.")},
  {"draftfolder",   ARG_DRAFTFOLDER, N_("FOLDER"), 0,
   N_("specify the folder for message drafts")},
  {"nodraftfolder", ARG_NODRAFTFOLDER, 0, 0,
   N_("undo the effect of the last --draftfolder option")},
  {"draftmessage" , ARG_DRAFTMESSAGE, N_("MSG"), 0,
   N_("invoke the draftmessage facility")},
  {"folder",        ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"file",          ARG_FILE, N_("FILE"), 0,
   N_("use FILE as the message draft")},
  {"editor",        ARG_EDITOR, N_("PROG"), 0,
   N_("set the editor program to use")},
  {"noedit",        ARG_NOEDIT, 0, 0,
   N_("suppress the initial edit")},
  {"form",          ARG_FORM, N_("FILE"), 0,
   N_("read format from given file")},
  {"whatnowproc",   ARG_WHATNOWPROC, N_("PROG"), 0,
   N_("* set the replacement for whatnow program")},
  {"nowhatnowproc", ARG_NOWHATNOWPROC, NULL, 0,
   N_("* ignore whatnowproc variable. Use standard `whatnow' shell instead.")},
  {"use",           ARG_USE, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("use draft file preserved after the last session") },
  {"nouse",         ARG_NOUSE, NULL, OPTION_HIDDEN, ""},
  {"license",       ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"build",         1, },
  {"file",          2, MH_OPT_ARG, "draftfile"},
  {"form",          2, MH_OPT_ARG, "formatfile"},
  {"draftfolder",   6, MH_OPT_ARG, "folder"},
  {"nodraftfolder", 3, },
  {"draftmessage",  6, },
  {"editor",        1, MH_OPT_ARG, "program"},
  {"noedit",        3, },
  {"whatnowproc",   2, MH_OPT_ARG, "program"},
  {"nowhatnowproc", 3, },
  { 0 }
};

struct mh_whatnow_env wh_env = { 0 };
const char *formfile;
static int initial_edit = 1;
static int build_only = 0; /* --build flag */
static int use_draft = 0;  /* --use flag */
static char *draftmessage = "new";

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARGP_KEY_INIT:
      wh_env.draftfolder = mh_global_profile_get ("Draft-Folder",
						  mu_folder_directory ());
      break;

    case ARG_BUILD:
      build_only = 1;
      break;
      
    case ARG_DRAFTFOLDER:
      wh_env.draftfolder = arg;
      break;
      
    case ARG_EDITOR:
      wh_env.editor = arg;
      break;
      
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_FORM:
      formfile = mh_expand_name (MHLIBDIR, arg, 0);
      break;

    case ARG_DRAFTMESSAGE:
      draftmessage = arg;
      break;

    case ARG_USE:
      use_draft = is_true (arg);
      draftmessage = "cur";
      break;

    case ARG_NOUSE:
      use_draft = 0;
      break;
      
    case ARG_FILE:
      wh_env.draftfile = mh_expand_name (NULL, arg, 0);
      break;
	
    case ARG_NODRAFTFOLDER:
      wh_env.draftfolder = NULL;
      break;

    case ARG_NOEDIT:
      initial_edit = 0;
      break;
      
    case ARG_WHATNOWPROC:
    case ARG_NOWHATNOWPROC:
      mh_opt_notimpl ("-[no]whatnowproc");
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
copy_message (mu_mailbox_t mbox, size_t n, const char *file)
{
  mu_message_t msg;
  mu_stream_t in;
  mu_stream_t out;
  int rc;
  size_t size;
  char *buffer;
  size_t bufsize, rdsize;
  
  mu_mailbox_get_message (mbox, n, &msg);
  mu_message_size (msg, &size);

  for (bufsize = size; bufsize > 0 && (buffer = malloc (bufsize)) == 0;
       bufsize /= 2)
    ;

  if (!bufsize)
    mh_err_memory (1);

  mu_message_get_stream (msg, &in);
  
  if ((rc = mu_file_stream_create (&out,
				file, MU_STREAM_RDWR|MU_STREAM_CREAT)) != 0
      || (rc = mu_stream_open (out)))
    {
      mu_error (_("cannot open output file \"%s\": %s"),
		file, mu_strerror (rc));
      free (buffer);
      return 1;
    }

  while (size > 0
	 && (rc = mu_stream_sequential_read (in, buffer, bufsize, &rdsize)) == 0
	 && rdsize > 0)
    {
      if ((rc = mu_stream_sequential_write (out, buffer, rdsize)) != 0)
	{
	  mu_error (_("error writing to \"%s\": %s"),
		    file, mu_strerror (rc));
	  break;
	}
      size -= rdsize;
    }

  mu_stream_close (out);
  mu_stream_destroy (&out, mu_stream_get_owner (out));
  
  return rc;
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

  if (build_only || !wh_env.draftfolder)
    wh_env.file = mh_expand_name (NULL, "draft", 0);
  else if (wh_env.draftfolder)
    {
      if (mh_draft_message (wh_env.draftfolder, draftmessage, &wh_env.file))
	return 1;
    }
  wh_env.draftfile = wh_env.file;

  switch (check_draft_disposition (&wh_env, use_draft))
    {
    case DISP_QUIT:
      exit (0);

    case DISP_USE:
      break;
	  
    case DISP_REPLACE:
      unlink (wh_env.draftfile);
  
      if (index < argc)
	{
	  static mh_msgset_t msgset;
	  static mu_mailbox_t mbox;
	  
	  mbox = mh_open_folder (mh_current_folder (), 0);
	  mh_msgset_parse (mbox, &msgset, argc - index, argv + index, "cur");
	  if (msgset.count != 1)
	    {
	      mu_error (_("only one message at a time!"));
	      return 1;
	    }
	  copy_message (mbox, msgset.list[0], wh_env.file);
	}
      else
	mh_comp_draft (formfile, "components", wh_env.file);
    }
  
  /* Exit immediately if --build is given */
  if (build_only)
    return 0;

  return mh_whatnow (&wh_env, initial_edit);
}
