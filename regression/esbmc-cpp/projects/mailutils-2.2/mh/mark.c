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

#include <mh.h>

const char *program_version = "mark (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH mark")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "[msgs...]";

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon")},
  {"sequence", ARG_SEQUENCE, N_("NAME"), 0,
   N_("specify sequence name to operate upon")},
  {"add", ARG_ADD, NULL, 0,
   N_("add messages to the sequence")},
  {"delete", ARG_DELETE, NULL, 0,
   N_("delete messages from the sequence")},
  {"list", ARG_LIST, NULL, 0,
   N_("list the sequences")},
  {"public", ARG_PUBLIC, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("create public sequence")},
  {"nopublic", ARG_NOPUBLIC, NULL, OPTION_HIDDEN, "" },
  {"zero", ARG_ZERO, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("empty the sequence before adding messages")},
  {"nozero", ARG_NOZERO, NULL, OPTION_HIDDEN, "" },
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  {NULL}
};

struct mh_option mh_option[] = {
  {"sequence", 1, },
  {"add", 1, }, 
  {"delete", 1, },
  {"list", 1, },
  {"public", 1, MH_OPT_BOOL },
  {"zero", 1, MH_OPT_BOOL },
  { NULL }
};

static int action;  /* Action to perform */
static int seq_flags = 0; /* Create public sequences;
			     Do not zero the sequence before addition */
static mu_list_t seq_list;  /* List of sequence names to operate upon */

static const char *mbox_dir;

static void
add_sequence (char *name)
{
  if (!seq_list && mu_list_create (&seq_list))
    {
      mu_error (_("cannot create sequence list"));
      exit (1);
    }
  mu_list_append (seq_list, name);
}

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_SEQUENCE:
      add_sequence (arg);
      break;

    case ARG_ADD:
    case ARG_DELETE:
    case ARG_LIST:
      action = key;
      break;
      
    case ARG_PUBLIC:
      if (is_true (arg))
	seq_flags &= ~SEQ_PRIVATE;
      else
	seq_flags |= SEQ_PRIVATE;
      break;
      
    case ARG_NOPUBLIC:
      seq_flags |= SEQ_PRIVATE;
      break;
      
    case ARG_ZERO:
      if (is_true (arg))
	seq_flags |= SEQ_ZERO;
      else
	seq_flags &= ~SEQ_ZERO;
      break;

    case ARG_NOZERO:
      seq_flags &= ~SEQ_ZERO;
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static int
action_add (void *item, void *data)
{
  mh_seq_add ((char *)item, (mh_msgset_t *)data, seq_flags);
  return 0;
}

static int
action_delete (void *item, void *data)
{
  mh_seq_delete ((char *)item, (mh_msgset_t *)data, seq_flags);
  return 0;
}

static int
action_list (void *item, void *data)
{
  char *name = item;
  const char *val;
  
  val = mh_seq_read (name, 0);
  if (val)
    printf ("%s: %s\n", name, val);
  else if ((val = mh_seq_read (name, SEQ_PRIVATE)))
    printf ("%s (%s): %s\n", name, _("private"), val);
  return 0;
}

static int
list_private (const char *name, const char *value, void *data)
{
  int nlen;
  
  if (memcmp (name, "atr-", 4))
    return 0;
  name += 4;

  nlen = strlen (name) - strlen (mbox_dir);
  if (nlen > 0 && strcmp (name + nlen, mbox_dir) == 0)
    {
      nlen--;
      printf ("%*.*s (%s): %s\n", nlen, nlen, name, _("private"), value);
    }
  return 0;
}

static int
list_public (const char *name, const char *value, void *data)
{
  printf ("%s: %s\n", name, value);
  return 0;
}

static void
list_all ()
{
  mh_global_sequences_iterate (list_public, NULL);
  mh_global_context_iterate (list_private, NULL);
}

int
main (int argc, char **argv)
{
  int index;
  mh_msgset_t msgset;
  mu_mailbox_t mbox;
  mu_url_t url;
  
  MU_APP_INIT_NLS ();
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  mbox = mh_open_folder (mh_current_folder (), 0);
  mu_mailbox_get_url (mbox, &url);
  mbox_dir = mu_url_to_string (url);
  if (memcmp (mbox_dir, "mh:", 3) == 0)
    mbox_dir += 3;
	
  argc -= index;
  argv += index;
  mh_msgset_parse (mbox, &msgset, argc, argv, "cur");
  mh_msgset_uids (mbox, &msgset);
  
  switch (action)
    {
    case ARG_ADD:
      if (!seq_list)
	{
	  mu_error (_("--add requires at least one --sequence argument"));
	  return 1;
	}
      mu_list_do (seq_list, action_add, (void *) &msgset);
      mh_global_save_state ();
      break;
      
    case ARG_DELETE:
      if (!seq_list)
	{
	  mu_error (_("--delete requires at least one --sequence argument"));
	  return 1;
	}
      mu_list_do (seq_list, action_delete, (void *) &msgset);
      mh_global_save_state ();
      break;
      
    case ARG_LIST:
      if (!seq_list)
	list_all ();
      else
	mu_list_do (seq_list, action_list, NULL);
      break;
    }

  return 0;
}
