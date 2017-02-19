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

/* MH forw command */

#include <mh.h>

const char *program_version = "forw (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH forw")"\v"
N_("Options marked with `*' are not yet implemented.\n\
Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[msgs]";

/* GNU options */
static struct argp_option options[] = {
  {"annotate",      ARG_ANNOTATE,      N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("add Forwarded: header to each forwarded message")},
  {"build",         ARG_BUILD,         0, 0,
   N_("build the draft and quit immediately")},
  {"draftfolder",   ARG_DRAFTFOLDER,   N_("FOLDER"), 0,
   N_("specify the folder for message drafts")},
  {"nodraftfolder", ARG_NODRAFTFOLDER, 0, 0,
   N_("undo the effect of the last --draftfolder option")},
  {"draftmessage" , ARG_DRAFTMESSAGE,  N_("MSG"), 0,
   N_("invoke the draftmessage facility")},
  {"folder",        ARG_FOLDER,        N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"editor",        ARG_EDITOR,        N_("PROG"), 0,
   N_("set the editor program to use")},
  {"noedit",        ARG_NOEDIT,        0, 0,
   N_("suppress the initial edit")},
  {"format",        ARG_FORMAT,        N_("BOOL"), 0, 
   N_("format messages")},
  {"noformat",      ARG_NOFORMAT,      NULL, 0,
   N_("undo the effect of the last --format option") },
  {"form",          ARG_FORM,          N_("FILE"), 0,
   N_("read format from given file")},
  {"filter",        ARG_FILTER,        N_("FILE"), 0,
  N_("use filter FILE to preprocess the body of the message") },
  {"nofilter",      ARG_NOFILTER,      NULL, 0,
   N_("undo the effect of the last --filter option") },
  {"inplace",       ARG_INPLACE,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("* annotate the message in place")},
  {"noinplace",     ARG_NOINPLACE,     0,          OPTION_HIDDEN, "" },
  {"mime",          ARG_MIME,          N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("use MIME encapsulation") },
  {"nomime",        ARG_NOMIME,        NULL, OPTION_HIDDEN, "" },
  {"width", ARG_WIDTH, N_("NUMBER"), 0, N_("Set output width")},
  {"whatnowproc",   ARG_WHATNOWPROC,   N_("PROG"), 0,
   N_("* set the replacement for whatnow program")},
  {"nowhatnowproc", ARG_NOWHATNOWPROC, NULL, 0,
   N_("* ignore whatnowproc variable, use standard `whatnow' shell instead")},
  {"use",           ARG_USE,           N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("use draft file preserved after the last session") },
  {"nouse",         ARG_NOUSE,         N_("BOOL"), OPTION_HIDDEN, "" },

  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},

  {NULL},
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"annotate",      1, MH_OPT_BOOL },
  {"build",         1, },
  {"form",          4, MH_OPT_ARG, "formatfile"},
  {"format",        5,  MH_OPT_ARG, "string"},
  {"draftfolder",   6, MH_OPT_ARG, "folder"},
  {"nodraftfolder", 3 },
  {"draftmessage",  6, },
  {"editor",        1, MH_OPT_ARG, "program"},
  {"noedit",        3, },
  {"filter",        2, MH_OPT_ARG, "program"},
  {"inplace",       1, MH_OPT_BOOL },
  {"whatnowproc",   2, MH_OPT_ARG, "program"},
  {"nowhatnowproc", 3 },
  {"mime",          2, MH_OPT_BOOL, NULL},
  {NULL}
};

enum encap_type {
  encap_clear,
  encap_mhl,
  encap_mime
};

static char *formfile;
struct mh_whatnow_env wh_env = { 0 };
static int initial_edit = 1;
static char *mhl_filter = NULL; /* --filter flag */
static int build_only = 0;      /* --build flag */
static int annotate = 0;        /* --annotate flag */
static enum encap_type encap = encap_clear; /* controlled by --format, --form
					       and --mime flags */
static int use_draft = 0;       /* --use flag */
static int width = 80;          /* --width flag */
static char *draftmessage = "new";

static mh_msgset_t msgset;
static mu_mailbox_t mbox;

static int
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARGP_KEY_INIT:
      wh_env.draftfolder = mh_global_profile_get ("Draft-Folder",
                                                   mu_folder_directory ());
      break;

    case ARG_ANNOTATE:
      annotate = is_true (arg);
      break;
      
    case ARG_BUILD:
      build_only = 1;
      break;

    case ARG_DRAFTFOLDER:
      wh_env.draftfolder = arg;
      break;

    case ARG_NODRAFTFOLDER:
      wh_env.draftfolder = NULL;
      break;
      
    case ARG_DRAFTMESSAGE:
      draftmessage = arg;
      break;

    case ARG_USE:
      use_draft = is_true (arg);
      break;

    case ARG_NOUSE:
      use_draft = 0;
      break;

    case ARG_WIDTH:
      width = strtoul (arg, NULL, 0);
      if (!width)
	{
	  argp_error (state, _("invalid width"));
	  exit (1);
	}
      break;

    case ARG_EDITOR:
      wh_env.editor = arg;
      break;
      
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_FORM:
      formfile = arg;
      break;

    case ARG_FORMAT:
      if (is_true (arg))
	{
	  encap = encap_mhl;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOFORMAT:
      if (encap == encap_mhl)
	encap = encap_clear;
      break;

    case ARG_FILTER:
      encap = encap_mhl;
      break;
	
    case ARG_MIME:
      if (is_true (arg))
	{
	  encap = encap_mime;
	  break;
	}
      /*FALLTHRU*/
    case ARG_NOMIME:
      if (encap == encap_mime)
	encap = encap_clear;
      break;
      
    case ARG_INPLACE:
      mh_opt_notimpl_warning ("-inplace");
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

struct format_data {
  int num;
  mu_stream_t stream;
  mu_list_t format;
};

/* State machine according to RFC 934:
   
      S1 ::   CRLF {CRLF} S1
            | "-" {"- -"} S2
            | c {c} S2

      S2 ::   CRLF {CRLF} S1
            | c {c} S2
*/

enum rfc934_state { S1, S2 };

static int
msg_copy (mu_message_t msg, mu_stream_t ostream)
{
  mu_stream_t istream;
  int rc;
  size_t n;
  char buf[512];
  enum rfc934_state state = S1;
  
  rc = mu_message_get_stream (msg, &istream);
  if (rc)
    return rc;
  mu_stream_seek (istream, 0, SEEK_SET);
  while (rc == 0
	 && mu_stream_sequential_read (istream, buf, sizeof buf, &n) == 0
	 && n > 0)
    {
      size_t start, i;
	
      for (i = start = 0; i < n; i++)
	switch (state)
	  {
	  case S1:
	    if (buf[i] == '-')
	      {
		rc = mu_stream_sequential_write (ostream, buf + start,
						 i - start + 1);
		if (rc)
		  return rc;
		rc = mu_stream_sequential_write (ostream, " -", 2);
		if (rc)
		  return rc;
		start = i + 1;
		state = S2;
	      }
	    else if (buf[i] != '\n')
	      state = S2;
	    break;
	      
	  case S2:
	    if (buf[i] == '\n')
	      state = S1;
	  }
      if (i > start)
	rc = mu_stream_sequential_write (ostream, buf + start, i  - start);
    }
  return rc;
}

void
format_message (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  struct format_data *fp = data;
  char *s;
  int rc;
  
  if (annotate)
    mu_list_append (wh_env.anno_list, msg);
  
  if (fp->num)
    {
      asprintf (&s, "\n------- Message %d\n", fp->num++);
      rc = mu_stream_sequential_write (fp->stream, s, strlen (s));
      free (s);
    }

  if (fp->format)
    rc = mhl_format_run (fp->format, width, 0, 0, msg, fp->stream);
  else
    rc = msg_copy (msg, fp->stream);
}

void
finish_draft ()
{
  int rc;
  mu_stream_t stream;
  mu_list_t format = NULL;
  struct format_data fd;
  char *str;
  
  if (!mhl_filter)
    {
      char *s = mh_expand_name (MHLIBDIR, "mhl.forward", 0);
      if (access (s, R_OK) == 0)
	mhl_filter = "mhl.forward";
      free (s);
    }

  if (mhl_filter)
    {
      char *s = mh_expand_name (MHLIBDIR, mhl_filter, 0);
      format = mhl_format_compile (s);
      if (!format)
	exit (1);
      free (s);
    }
  
  if ((rc = mu_file_stream_create (&stream,
				   wh_env.file,
				   MU_STREAM_WRITE|MU_STREAM_CREAT)) != 0
      || (rc = mu_stream_open (stream)))
    {
      mu_error (_("cannot open output file `%s': %s"),
		wh_env.file, mu_strerror (rc));
      exit (1);
    }

  mu_stream_seek (stream, 0, SEEK_END);

  if (annotate)
    {
      wh_env.anno_field = "Forwarded";
      mu_list_create (&wh_env.anno_list);
    }
  
  if (encap == encap_mime)
    {
      mu_url_t url;
      const char *mbox_path;
      const char *p;
      size_t i;
      
      mu_mailbox_get_url (mbox, &url);
      
      mbox_path = mu_url_to_string (url);
      if (memcmp (mbox_path, "mh:", 3) == 0)
	mbox_path += 3;
      asprintf (&str, "#forw [] +%s", mbox_path);
      rc = mu_stream_sequential_write (stream, str, strlen (str));
      free (str);
      for (i = 0; rc == 0 && i < msgset.count; i++)
	{
          mu_message_t msg;
	  size_t num;
		  
	  mu_mailbox_get_message (mbox, msgset.list[i], &msg);
	  if (annotate)
	    mu_list_append (wh_env.anno_list, msg);
	  mh_message_number (msg, &num);
          p = mu_umaxtostr (0, num);
          rc = mu_stream_sequential_write (stream, p, strlen (p));
	}
    }
  else
    {
      str = "\n------- ";
      rc = mu_stream_sequential_write (stream, str, strlen (str));

      if (msgset.count == 1)
	{
	  fd.num = 0;
	  str = (char*) _("Forwarded message\n");
	}
      else
	{
	  fd.num = 1;
	  str = (char*) _("Forwarded messages\n");
	}
  
      rc = mu_stream_sequential_write (stream, str, strlen (str));
      fd.stream = stream;
      fd.format = format;
      rc = mh_iterate (mbox, &msgset, format_message, &fd);
      
      str = "\n------- ";
      rc = mu_stream_sequential_write (stream, str, strlen (str));
      
      if (msgset.count == 1)
	str = (char*) _("End of Forwarded message");
      else
	str = (char*) _("End of Forwarded messages");
      
      rc = mu_stream_sequential_write (stream, str, strlen (str));
    }
  
  rc = mu_stream_sequential_write (stream, "\n\n", 2);
  mu_stream_close (stream);
  mu_stream_destroy (&stream, mu_stream_get_owner (stream));
}

int
main (int argc, char **argv)
{
  int index, rc;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  argc -= index;
  argv += index;

  mbox = mh_open_folder (mh_current_folder (), 0);
  mh_msgset_parse (mbox, &msgset, argc, argv, "cur");
  
  if (build_only || !wh_env.draftfolder)
    wh_env.file = mh_expand_name (NULL, "draft", 0);
  else 
    {
      if (mh_draft_message (NULL, draftmessage, &wh_env.file))
	return 1;
    }
  wh_env.draftfile = wh_env.file;

  switch (build_only ?
	    DISP_REPLACE : check_draft_disposition (&wh_env, use_draft))
    {
    case DISP_QUIT:
      exit (0);

    case DISP_USE:
      break;
	  
    case DISP_REPLACE:
      unlink (wh_env.draftfile);
      mh_comp_draft (formfile, "forwcomps", wh_env.file);
      finish_draft ();
    }
  
  /* Exit immediately if --build is given */
  if (build_only)
    {
      if (strcmp (wh_env.file, wh_env.draftfile))
	rename (wh_env.file, wh_env.draftfile);
      return 0;
    }
  
  rc = mh_whatnow (&wh_env, initial_edit);

  mu_mailbox_sync (mbox);
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return rc;
}
