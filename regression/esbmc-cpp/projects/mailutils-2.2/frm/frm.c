/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
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

#include <frm.h>

static int show_summary;   /* Summarize the number of messages by message
			      status in each mailbox. -S option */
static int be_quiet;       /* Quiet mode. -q option. */
static int show_query;     /* Additional flag toggled by -q to display 
			      a one-line summary for each mailbox */
static int align = 0;      /* Tidy mode. -t option. */

#define IS_READ 0x001
#define IS_OLD  0x010
#define IS_NEW  0x100
static int select_attribute;

const char *program_version = "frm (" PACKAGE_STRING ")";
static char doc[] = N_("GNU frm -- display From: lines.");

static struct attr_tab {
  char *name;      /* Attribute name */
  int code;        /* Corresponding IS_.* flag */
  size_t len;      /* Minimum abbreviation length */
} attr_tab[] = {
  { "new", IS_NEW, 0 },
  { "old", IS_OLD, 0 },
  { "unread", IS_OLD, 0 },
  { "read", IS_READ, 0 },
  { NULL }
};


/* Attribute table handling */

/* Prepares the table for use. Computes minimum abbreviation lengths. */
static void
prepare_attrs (void)
{
  struct attr_tab *p, *q;

  for (p = attr_tab; p->name; p++)
    {
      const char *name = p->name;
      size_t len = strlen (name);
      size_t n = 1;

      for (q = attr_tab; q->name; q++)
	{
	  if (p != q)
	    {
	      const char *str = q->name;
	      size_t slen = strlen (str);

	      if (memcmp (name, str, n) == 0)
		{
		  for (n++;
		       memcmp (name, str, n) == 0
			 && n < len
			 && n < slen; n++)
		    ;
		  
		  q->len = n < slen ? n : slen;
		}
	    }
	}
      p->len = n < len ? n : len;
    }
}

/* Translates the textual status representation to the corresponding
   IS_.* flag */
static int
decode_attr (char *arg)
{
  struct attr_tab *p;
  int len = strlen (arg);
  int pretendents = 0;
  
  for (p = attr_tab; p->name; p++)
    {
      const char *str = p->name;
      if (str[0] == arg[0])
	{
	  if (len < p->len)
	    pretendents++;
	  else if (len > strlen (str))
	    continue;
	  if (memcmp (str, arg, p->len) == 0)
	    return p->code;
	}
    }
  if (pretendents)
    mu_error (_("%s: ambiguous abbreviation"), arg);
  else
    mu_error (_("%s: unknown attribute"), arg);
  return 0;
}



static struct argp_option options[] = {
  {"debug",  'd', NULL,   0, N_("enable debugging output"), 0},
  {"field",  'f', N_("NAME"), 0, N_("header field to display"), 0},
  {"to",     'l', NULL,   0, N_("include the To: information"), 0},
  {"number", 'n', NULL,   0, N_("display message numbers"), 0},
  {"Quiet",  'Q', NULL,   0, N_("do not display headers"), 0},
  {"query",  'q', NULL,   0, N_("print a message if the mailbox contains some unread mail"), 0},
  {"summary",'S', NULL,   0, N_("print a summary of messages"), 0},
  {"status", 's', N_("STATUS"), 0,
   /* TRANSLATORS: Please do *not* translate the words "new", "unread",
      "old" and "read". They are keywords. */
   N_("select messages with the specific attribute:"
      " new, unread, old (same as unread) or read (or any unambiguous"
      " abbreviation of these)"),
   0},
  {"align",  't', NULL,   0, N_("tidy mode: align subject lines"), 0},
  {0, 0, 0, 0}
};

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'd':
      frm_debug++;
      break;

    case 'f':
      show_field = arg;
      align = 0;
      break;

    case 'l':
      show_to = 1;
      break;

    case 'n':
      show_number = 1;
      break;

    case 'Q':
      /* Very silent.  */
      be_quiet += 2;
      break;

    case 'q':
      be_quiet++;
      show_query = 1;
      break;

    case 'S':
      show_summary = 1;
      break;

    case 's':
      select_attribute = decode_attr (arg);
      break;
      
    case 't':
      align = 1;
      break;

    case ARGP_KEY_FINI:
      {
	size_t s;
	if (align && (s = util_getcols ()))
	  init_output (s);
	else
	  init_output (0);
      }
      break;
      
    default: 
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {
  options,
  parse_opt,
  N_("[URL ...]"),
  doc,
  NULL,
  NULL, NULL
};

static const char *frm_argp_capa[] = {
  "common",
  "debug",
  "license",
  "mailbox",
  "locking",
  NULL
};


static struct
{
  size_t new;
  size_t read;
  size_t unread;
} counter;


static int selected;

static int
frm_select (size_t index, mu_message_t msg)
{
  mu_header_t hdr = NULL;
  mu_attribute_t attr = NULL;
  
  mu_message_get_attribute (msg, &attr);
  mu_message_get_header (msg, &hdr);

  if (mu_attribute_is_read (attr))
    counter.read++;
  else if (mu_attribute_is_seen (attr))
    counter.unread++;
  else if (mu_attribute_is_recent (attr))
    counter.new++;

  if (((select_attribute & IS_READ) && (!mu_attribute_is_read (attr)))
      || ((select_attribute & IS_NEW) && (!mu_attribute_is_recent (attr)))
      || ((select_attribute & IS_OLD) && (!mu_attribute_is_seen (attr))))
    return 0;
  
  if (select_attribute)
    selected++;

  return !be_quiet;
}


/* This is a clone of the elm program call "frm".  It is a good example on
   how to use the observable(callback) of libmailbox.  "frm" has to
   be very interactive, it is not possible to call mu_mailbox_messages_count()
   and wait for the scanning to finish before displaying.  As soon as the scan
   finds a new message we want to know about it. This is done by registering
   an observable type MU_MAILBOX_MSG_ADD. The rest is formatting code.  */

int
frm (char *mailbox_name)
{
  size_t total;
  int status;

  selected = 0;
  frm_scan (mailbox_name, frm_select, &total);
  
  if (show_summary)
    {
      if (total == 0)
	printf (_("Folder contains no messages."));
      else
	{
	  char *delim = "";
      	  
	  printf (_("Folder contains "));
	
	  if (counter.new)
	    {
	      printf (ngettext ("%lu new message",
 				"%lu new messages",
				counter.new),
		      (u_long) counter.new);
	      delim = ", ";
	    }
	  
	  if (counter.unread)
	    {
	      printf ("%s", delim);
	      
	      printf (ngettext ("%lu unread message",
				"%lu unread messages",
				counter.unread),
		      (u_long) counter.unread);
	      delim = ", ";
	    }
	  
	  if (counter.read)
	    {
	      printf ("%s", delim);
	      
	      printf (ngettext ("%lu read message",
				"%lu read messages",
				counter.read),
		      (u_long) counter.read);
	    }
	  /* TRANSLATORS: This dot finishes the sentence

              "Folder contains XXX messages."

	      Leave it as it is unless your language requires to reorder
	      the parts of speach in the message
	  */
	  printf (_("."));
	}
      printf ("\n");
    }
  else if (show_query)
    {
      if (total > 0)
	printf (_("There are messages in that folder.\n"));
      else
	printf (_("No messages in that folder!\n"));
    }
  
  /* EXIT STATUS
     Frm returns a zero status ("true") if messages matching `status' are
     present.  Frm returns 1 if no messages matching `status' are present,
     but there are some messages, returns 2 if there are no messages at
     all, or returns 3 if an error occurred. */
  
  if (selected)
    status = 0;
  else if (total > 0)
    status = 1;
  else
    status = 2;

  return status;
}

int
main (int argc, char **argv)
{
  int c;
  int status = 0;

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  prepare_attrs ();
  
  /* register the formats.  */
  mu_register_all_mbox_formats ();
#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif
  
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, frm_argp_capa, NULL, argc, argv, 0, &c, NULL))
    exit (1);

  /* have an argument */
  if (c == argc)
    status = frm (NULL);
  else if (c + 1 == argc)
    status = frm (argv[c]);
  else
    for (; c < argc; c++)
      {
	printf ("%s:\n", argv[c]);
	status = frm (argv[c]);
      }

  return status;
}
