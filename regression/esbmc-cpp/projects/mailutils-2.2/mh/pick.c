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

/* MH pick command */

#include <mh.h>
#include <regex.h>
#include <pick.h>
#include <pick-gram.h>
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
#include <obstack.h>

const char *program_version = "pick (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH pick")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[messages]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to operate upon"), 0},

  {N_("Specifying search patterns:"), 0,  NULL, OPTION_DOC,  NULL, 0},
  {"component", ARG_COMPONENT, N_("FIELD"), 0,
   N_("search the named header field"), 1},
  {"pattern", ARG_PATTERN, N_("STRING"), 0,
   N_("set pattern to look for"), 1},
  {"search",  0, NULL, OPTION_ALIAS, NULL, 1},
  {"cflags",  ARG_CFLAGS,  N_("STRING"), 0,
   N_("flags controlling the type of regular expressions. STRING must consist of one or more of the following letters: B=basic, E=extended, I=ignore case, C=case sensitive. Default is \"EI\". The flags remain in effect until the next occurrence of --cflags option. The option must occur right before --pattern or --component option (or its alias).") },
  {"cc",      ARG_CC,      N_("STRING"), 0,
   N_("same as --component cc --pattern STRING"), 1},
  {"date",    ARG_DATE,    N_("STRING"), 0,
   N_("same as --component date --pattern STRING"), 1},
  {"from",    ARG_FROM,    N_("STRING"), 0,
   N_("same as --component from --pattern STRING"), 1},
  {"subject", ARG_SUBJECT, N_("STRING"), 0,
   N_("same as --component subject --pattern STRING"), 1},
  {"to",      ARG_TO,      N_("STRING"), 0,
   N_("same as --component to --pattern STRING"), 1},

  {N_("Date constraint operations:"), 0,  NULL, OPTION_DOC, NULL, 1},
  {"datefield",ARG_DATEFIELD, N_("STRING"), 0,
   N_("search in the named date header field (default is `Date:')"), 2},
  {"after",    ARG_AFTER,     N_("DATE"), 0,
   N_("match messages after the given date"), 2},
  {"before",   ARG_BEFORE,    N_("DATE"), 0,
   N_("match messages before the given date"), 2},

  {N_("Logical operations and grouping:"), 0, NULL, OPTION_DOC, NULL, 2},
  {"and",     ARG_AND,    NULL, 0,
   N_("logical AND (default)"), 3 },
  {"or",      ARG_OR,     NULL, 0,
   N_("logical OR"), 3 },
  {"not",     ARG_NOT,    NULL, 0,
   N_("logical NOT"), 3},
  {"lbrace",  ARG_LBRACE, NULL, 0,
   N_("open group"), 3},
  {"(",       0, NULL, OPTION_ALIAS, NULL, 3},
  {"rbrace",  ARG_RBRACE, NULL, 0,
   N_("close group"), 3},
  {")",       0, NULL, OPTION_ALIAS, NULL, 3},

  {N_("Operations over the selected messages:"), 0, NULL, OPTION_DOC, NULL, 3},
  {"list",   ARG_LIST,       N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("list the numbers of the selected messages (default)"), 4},
  {"nolist", ARG_NOLIST,     NULL, OPTION_HIDDEN, "", 4 },
  {"sequence", ARG_SEQUENCE,  N_("NAME"), 0,
   N_("add matching messages to the given sequence"), 4},
  {"public", ARG_PUBLIC, N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("create public sequence"), 4},
  {"nopublic", ARG_NOPUBLIC, NULL, OPTION_HIDDEN, "", 4 },
  {"zero",     ARG_ZERO,     N_("BOOL"), OPTION_ARG_OPTIONAL,
   N_("empty the sequence before adding messages"), 4},
  {"nozero", ARG_NOZERO, NULL, OPTION_HIDDEN, "", 4 },
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},
  {NULL},
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"component", 1,  0, "field" },
  {"pattern",   1,  0, "pattern" },
  {"search",    1,  0, "pattern" },
  {"cc",        1,  0, "pattern" },
  {"date",      1,  0, "pattern" },
  {"from",      1,  0, "pattern" },
  {"subject",   1,  0, "pattern" },
  {"to",        1,  0, "pattern" },
  {"datefield", 1,  0, "field" },
  {"after",     1,  0, "date" },
  {"before",    1,  0, "date"},
  {"and",       1,  0, NULL },
  {"or",        1,  0, NULL }, 
  {"not",       1,  0, NULL },
  {"lbrace",    1,  0, NULL },
  {"rbrace",    1,  0, NULL },

  {"list",      1,  MH_OPT_BOOL, },
  {"sequence",  1,  0, NULL },
  {"public",    1,  MH_OPT_BOOL },
  {"zero",      1,  MH_OPT_BOOL },
  {NULL}
};

static int list = 1;
static int seq_flags = 0; /* Create public sequences;
			     Do not zero the sequence before addition */
static mu_list_t seq_list;  /* List of sequence names to operate upon */

static mu_list_t lexlist;   /* List of input tokens */

static struct obstack msgno_stk; /* Stack of selected message numbers */
static size_t msgno_count;       /* Number of items on the stack */

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
  char *s, *p;
  
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;

    case ARG_SEQUENCE:
      add_sequence (arg);
      list = 0;
      break;

    case ARG_LIST:
      list = is_true (arg);
      break;

    case ARG_NOLIST:
      list = 0;
      break;

    case ARG_COMPONENT:
      pick_add_token (&lexlist, T_COMP, arg);
      break;
      
    case ARG_PATTERN:
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_CC:
      pick_add_token (&lexlist, T_COMP, "cc");
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_DATE:           
      pick_add_token (&lexlist, T_COMP, "date");
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_FROM:           
      pick_add_token (&lexlist, T_COMP, "from");
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_SUBJECT:        
      pick_add_token (&lexlist, T_COMP, "subject");
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_TO:
      pick_add_token (&lexlist, T_COMP, "to");
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_DATEFIELD:
      pick_add_token (&lexlist, T_DATEFIELD, arg);
      break;
      
    case ARG_AFTER:
      pick_add_token (&lexlist, T_AFTER, NULL);
      pick_add_token (&lexlist, T_STRING, arg);
      break;
      
    case ARG_BEFORE:
      pick_add_token (&lexlist, T_BEFORE, NULL);
      pick_add_token (&lexlist, T_STRING, arg);
      break;
	
    case ARG_AND:
      pick_add_token (&lexlist, T_AND, NULL);
      break;
      
    case ARG_OR:
      pick_add_token (&lexlist, T_OR, NULL);
      break;
      
    case ARG_NOT:
      pick_add_token (&lexlist, T_NOT, NULL);
      break;

    case ARG_LBRACE:
      pick_add_token (&lexlist, T_LBRACE, NULL);
      break;
      
    case ARG_RBRACE:
      pick_add_token (&lexlist, T_RBRACE, NULL);
      break;

    case ARG_CFLAGS:
      pick_add_token (&lexlist, T_CFLAGS, arg);
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
	
    case ARGP_KEY_ERROR:
      s = state->argv[state->next - 1];
      if (memcmp (s, "--", 2))
	{
	  argp_error (state, _("invalid option -- %s"), s);
	  exit (1);
	}
      p = strchr (s, '=');
      if (p)
	*p++ = 0;
	
      pick_add_token (&lexlist, T_COMP, s + 2);

      if (!p)
	{
	  if (state->next == state->argc)
	    {
	      mu_error (_("invalid option -- %s"), s);
	      exit (1);
	    }
	  p = state->argv[state->next++];
	}
      
      pick_add_token (&lexlist, T_STRING, p);
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
pick_message (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  if (pick_eval (msg))
    {
      mh_message_number (msg, &num);
      if (list)
	printf ("%s\n", mu_umaxtostr (0, num));
      if (seq_list)
	{
	  obstack_grow (&msgno_stk, &num, sizeof (num));
	  msgno_count++;
	}
    }
}

static int
action_add (void *item, void *data)
{
  mh_seq_add ((char *)item, (mh_msgset_t *)data, seq_flags);
  return 0;
}

/* NOTICE: For the compatibility with the RAND MH we have to support
   the following command line syntax:

       --FIELD STRING

   where `FIELD' may be any string and which is equivalent to
   `--field FIELD --pattern STRING'. Obviously this is in conflict
   with the usual GNU long options paradigm which requires that any
   unrecognized long option produce an error. Unfortunately, mh-pick.el
   relies heavily on this syntax, so it can't be simply removed.
   The approach taken here allows to properly recognize such syntax,
   however it has an undesirable side effect: due to the specifics of
   the underlying arpg library the --help and --usage options get
   disabled. To make them work as well, the following approach is
   taken: the mh-compatible syntax gets enabled only if the file
   descriptor of stdin is not connected to a terminal, which is true
   when invoked from mh-pick.el module. Otherwise, it is disabled
   and the standard GNU long option syntax is in force. */
int
main (int argc, char **argv)
{
  int status;
  int index;
  mu_mailbox_t mbox;
  mh_msgset_t msgset;
  int flags;

  flags = mh_interactive_mode_p () ? 0 : ARGP_NO_ERRS;
  MU_APP_INIT_NLS ();
  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, flags, options, mh_option,
		 args_doc, doc, opt_handler, NULL, &index);
  if (pick_parse (lexlist))
    return 1;

  mbox = mh_open_folder (mh_current_folder (), 0);

  argc -= index;
  argv += index;

  if (seq_list)
    obstack_init (&msgno_stk);
  
  mh_msgset_parse (mbox, &msgset, argc, argv, "all");
  status = mh_iterate (mbox, &msgset, pick_message, NULL);

  if (seq_list)
    {
      mh_msgset_t msgset;
      msgset.count = msgno_count;
      msgset.list = obstack_finish (&msgno_stk);
      mu_list_do (seq_list, action_add, (void*) &msgset);
    }

  mh_global_save_state ();
  return status;
}
  
