/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

#include "readmsg.h"

static int
addset (int **set, int *n, unsigned val)
{
  int *tmp;
  tmp = realloc (*set, (*n + 1) * sizeof (**set));
  if (tmp == NULL)
    {
      if (*set)
        free (*set);
      *n = 0;
      *set = NULL;
      return ENOMEM;
    }
  *set = tmp;
  (*set)[*n] = val;
  (*n)++;
  return 0;
}

static int
is_number (const char *s)
{
  int result = 1;
  if (*s == '\0')
    result = 0;
  for (; *s; s++)
    {
      if (!mu_isdigit ((unsigned char)*s))
	{
	  result = 0;
	  break;
	}
    }
  return result;
}

/*
  According to ELM readmsg(1):

  1. A lone ``*'' means select all messages in the mailbox.

  2. A list of message numbers may be specified.  Values of ``0'' and
  ``$'' in the list both mean the last message in the mailbox.  For
  example:

  readmsg 1 3 0

  extracts three messages from the folder: the first, the third, and
  the last.

  3. Finally, the selection may be some text to match.  This will
  select a mail message which exactly matches the specified text. For
  example,

  readmsg staff meeting

  extracts the message which contains the words ``staff meeting.''
  Note that it will not match a message containing ``Staff Meeting'' -
  the matching is case sensitive.  Normally only the first message
  which matches the pattern will be printed.  The -a option discussed
  in a moment changes this.
*/

int
msglist (mu_mailbox_t mbox, int show_all, int argc, char **argv,
	 int **set, int *n)
{
  int i = 0;
  size_t total = 0;

  mu_mailbox_messages_count (mbox, &total);

  for (i = 0; i < argc; i++)
    {
      /* 1. A lone ``*'' means select all messages in the mailbox. */
      if (!strcmp (argv[i], "*"))
	{
	  size_t j;
	  /* all messages */
	  for (j = 1; j <= total; j++)
	    addset (set, n, j);
	  j = argc + 1;
	}
      /* 2. A list of message numbers may be specified.  Values of
	 ``0'' and ``$'' in the list both mean the last message in the
	 mailbox. */
      else if (!strcmp (argv[i], "$") || !strcmp (argv[i], "0"))
	{
	  addset (set, n, total);
	}
      /* 3. Finally, the selection may be some text to match.  This
	 will select a mail message which exactly matches the
	 specified text. */
      else if (!is_number (argv[i]))
	{
	  size_t j;
	  int found = 0;
	  for (j = 1; j <= total; j++)
	    {
	      char buf[128];
	      size_t len = 0;
	      mu_off_t offset = 0;
	      mu_message_t msg = NULL;
	      mu_stream_t stream = NULL;

	      mu_mailbox_get_message (mbox, j, &msg);
	      mu_message_get_stream (msg, &stream);
	      while (mu_stream_readline (stream, buf, sizeof buf,
					 offset, &len) == 0 && len > 0)
		{
		  if (strstr (buf, argv[i]) != NULL)
		    {
		      addset (set, n, j);
		      found = 1;
		      break;
		    }
		  offset += len;
		}
	      mu_stream_destroy (&stream, NULL);
	      if (found && !show_all)
		break;
	    }
	}
      else if (mu_isdigit (argv[i][0]))
	{
	  /* single message */
	  addset (set, n, strtol (argv[i], NULL, 10));
	}
    }

  return 0;
}
