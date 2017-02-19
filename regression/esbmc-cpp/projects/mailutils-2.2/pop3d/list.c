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
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "pop3d.h"

/* Displays the size of message number arg or all messages (if no arg) */

int
pop3d_list (char *arg)
{
  size_t mesgno;
  mu_message_t msg = NULL;
  mu_attribute_t attr = NULL;
  size_t size = 0;
  size_t lines = 0;

  if (state != TRANSACTION)
    return ERR_WRONG_STATE;

  if (strchr (arg, ' ') != NULL)
    return ERR_BAD_ARGS;

  if (strlen (arg) == 0)
    {
      size_t total = 0;
      pop3d_outf ("+OK\r\n");
      mu_mailbox_messages_count (mbox, &total);
      for (mesgno = 1; mesgno <= total; mesgno++)
	{
	  mu_mailbox_get_message (mbox, mesgno, &msg);
	  mu_message_get_attribute (msg, &attr);
	  if (!pop3d_is_deleted (attr))
	    {
	      mu_message_size (msg, &size);
	      mu_message_lines (msg, &lines);
	      pop3d_outf ("%s %s\r\n", 
                          mu_umaxtostr (0, mesgno), 
                          mu_umaxtostr (1, size + lines));
	    }
	}
      pop3d_outf (".\r\n");
    }
  else
    {
      mesgno = strtoul (arg, NULL, 10);
      if (mu_mailbox_get_message (mbox, mesgno, &msg) != 0)
	return ERR_NO_MESG;
      mu_message_get_attribute (msg, &attr);
      if (pop3d_is_deleted (attr))
	return ERR_MESG_DELE;
      mu_message_size (msg, &size);
      mu_message_lines (msg, &lines);
      pop3d_outf ("+OK %s %s\r\n", 
                  mu_umaxtostr (0, mesgno),
                  mu_umaxtostr (1, size + lines));
    }

  return OK;
}
