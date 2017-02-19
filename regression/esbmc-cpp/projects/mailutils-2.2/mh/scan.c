/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

/* MH scan command */

#include <mh.h>
#ifdef HAVE_TERMCAP_H
# include <termcap.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <mailutils/observer.h>

const char *program_version = "scan (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH scan")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = N_("[+folder] [msgs]");

/* GNU options */
static struct argp_option options[] = {
  {"folder",  ARG_FOLDER, N_("FOLDER"), 0,
   N_("specify folder to scan")},
  {"clear",   ARG_CLEAR, N_("BOOL"),   OPTION_ARG_OPTIONAL,
   N_("clear screen after displaying the list")},
  {"noclear", ARG_NOCLEAR, NULL, OPTION_HIDDEN, ""},
  {"form",    ARG_FORM, N_("FILE"),   0,
   N_("read format from given file")},
  {"format",  ARG_FORMAT, N_("FORMAT"), 0,
   N_("use this format string")},
  {"header",  ARG_HEADER, N_("BOOL"),   OPTION_ARG_OPTIONAL,
   N_("display header")},
  {"width",   ARG_WIDTH, N_("NUMBER"), 0,
   N_("set output width")},
  {"reverse", ARG_REVERSE, N_("BOOL"),   OPTION_ARG_OPTIONAL,
   N_("list messages in reverse order")},
  {"noreverse", ARG_NOREVERSE, NULL, OPTION_HIDDEN, ""},
  {"file",    ARG_FILE, N_("FILE"),   0,
   N_("[not yet implemented]")},
  
  {"license", ARG_LICENSE, 0,        0,
   N_("display software license"), -1},

  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"clear",   1,  MH_OPT_BOOL},
  {"form",    4,  MH_OPT_ARG, "formatfile"},
  {"format",  5,  MH_OPT_ARG, "string"},
  {"header",  3,  MH_OPT_BOOL},
  {"width",   1,  MH_OPT_ARG, "number"},
  {"reverse", 1,  MH_OPT_BOOL},
  {"file",    4,  MH_OPT_ARG, "file"},
  { 0 }
};

static int clear;
static char *format_str = mh_list_format;

static int width = 80;
static int reverse;
static int header;

static mh_format_t format;

static mh_msgset_t msgset;

void list_message (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data);
void print_header (mu_mailbox_t mbox);
void clear_screen (void);

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FOLDER: 
      mh_set_current_folder (arg);
      break;
      
    case ARG_CLEAR:
      clear = is_true(arg);
      break;

    case ARG_NOCLEAR:
      clear = 0;
      break;
      
    case ARG_FORM:
      mh_read_formfile (arg, &format_str);
      break;
      
    case ARG_FORMAT:
      format_str = arg;
      break;
      
    case ARG_HEADER:
      header = is_true(arg);
      break;
      
    case ARG_WIDTH:
      width = strtoul (arg, NULL, 0);
      if (!width)
	{
	  argp_error (state, _("invalid width"));
	  exit (1);
	}
      break;
      
    case ARG_REVERSE:
      reverse = is_true(arg);
      break;

    case ARG_NOREVERSE:
      reverse = 0;
      break;
      
    case ARG_FILE:
      mh_opt_notimpl ("-file");
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Observable Action this is called at every message discover.  */
static int
action (mu_observer_t o, size_t type, void *data, void *action_data)
{
  static int counter;
  mu_mailbox_t mbox;
  mu_message_t msg = NULL;
  size_t num;

  if (type == MU_EVT_MESSAGE_ADD)
    {
      mbox = mu_observer_get_owner (o);
      counter++;
      mu_mailbox_get_message (mbox, counter, &msg);
      mh_message_number (msg, &num);
      list_message (mbox, msg, num, NULL);
    }
  return 0;
}

int
main (int argc, char **argv)
{
  int index;
  mu_mailbox_t mbox;
  int status;
  size_t total = 0;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, &index);

  if (mh_format_parse (format_str, &format))
    {
      mu_error (_("Bad format string"));
      exit (1);
    }

  mbox = mh_open_folder (mh_current_folder (), 0);

  argc -= index;
  argv += index;
  if ((argc == 0 || strcmp (argv[0], "all") == 0) && !reverse)
    {
      /* Fast approach */
      mu_observer_t observer;
      mu_observable_t observable;
        
      print_header (mbox);
      
      mu_observer_create (&observer, mbox);
      mu_observer_set_action (observer, action, mbox);
      mu_mailbox_get_observable (mbox, &observable);
      mu_observable_attach (observable, MU_EVT_MESSAGE_ADD, observer);

      status = mu_mailbox_scan (mbox, 1, &total);
    }
  else
    {
      mu_mailbox_messages_count (mbox, &total);
      mh_msgset_parse (mbox, &msgset, argc, argv, "all");

      if (reverse)
	mh_msgset_reverse (&msgset);

      print_header (mbox);
      status = mh_iterate (mbox, &msgset, list_message, NULL);
    }

  if (total == 0)
    {
      mu_url_t url = NULL;

      mu_mailbox_get_url (mbox, &url);
      mu_error (_("no messages in %s"), mu_url_to_string (url));
    }

  clear_screen ();
  mh_global_save_state ();
  
  return status;
}

void
print_header (mu_mailbox_t mbox)
{
  if (header)
    {
      mu_url_t url = NULL;
      char datestr[64];
      time_t t;
	  
      mu_mailbox_get_url (mbox, &url);
      time (&t);
      strftime (datestr, sizeof datestr, "%c", localtime (&t));
      printf (_("Folder %s  %s\n"), mu_url_to_string (url), datestr);
    }
}

#ifdef HAVE_TERMCAP_H
/* A subroutine for tputs */
int
putstdout(char c)
{
  return putc(c, stdout);
}
#endif

void
clear_screen ()
{
  if (clear)
    {
#ifdef HAVE_TERMCAP_H
      if (isatty (1))
	{
	  char termcap_buf[1024];
	  char *buffer = termcap_buf;
	  char *termname;
      
	  if ((termname = getenv("TERM")) == NULL)
	    /* No terminal; Try ansi */
	    termname = "ansi";
      
	  if (tgetent(termcap_buf, termname) == 1)
	    {
	      char *clr = tgetstr ("cl", &buffer);
	      if (clr)
		{
		  tputs(clr, 1, (int (*)())putstdout);
		  return;
		}
	    }
	}
#endif
      /* Fall back to formfeed */
      fprintf (stdout, "\f");
    }
}

void
list_message (mu_mailbox_t mbox, mu_message_t msg, size_t num, void *data)
{
  char *buffer;
  int len;
  
  mh_format (&format, msg, num, width, &buffer);
  printf ("%s", buffer);
  len = strlen (buffer);
  if (len > 0 && buffer[len-1] != '\n')
    printf("\n");
  free (buffer);
}
