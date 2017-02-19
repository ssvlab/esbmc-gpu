/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2007, 2010 Free Software
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

#include "mail.h"

/* Simple summary dysplaying a blurb on the name of the
   mailbox and how many new:deleted:read messages.
   The side effect is that it sets the cursor
   to the newest or read message number.  */
int
mail_summary (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED)
{
  mu_message_t msg;
  mu_attribute_t attr;
  size_t msgno;
  size_t count = 0;
  int mseen = 0, mnew = 0, mdelete = 0;
  int first_new = 0, first_unread = 0;

  mu_mailbox_messages_count (mbox, &count);
  for (msgno = 1; msgno <= count; msgno++)
    {
      if ((mu_mailbox_get_message (mbox, msgno, &msg) == 0)
	  && (mu_message_get_attribute (msg, &attr) == 0))
	    {
	      int deleted = mu_attribute_is_deleted (attr);

	      if (deleted)
		mdelete++;
	      if (mu_attribute_is_seen (attr) && ! mu_attribute_is_read (attr))
		{
		  mseen++;
		  if (!deleted && !first_unread)
		    first_unread = msgno;
		}
	      if (mu_attribute_is_recent (attr))
		{
		  mnew++;
		  if (!deleted && !first_new)
		    first_new = msgno;
		}
	}
    }

  /* Print the mailbox name.  */
  {
    mu_url_t url = NULL;
    mu_mailbox_get_url (mbox, &url);
    printf("\"%s\": ", util_url_to_string (url));
  }
  printf (ngettext ("%d message", "%d messages", count), count);
  if (mnew > 0)
    printf (ngettext (" %d new", " %d new", mnew), mnew);
  if (mseen > 0)
    printf (ngettext (" %d unread", " %d unread", mseen), mseen);
  if (mdelete > 0)
    printf (ngettext (" %d deleted", " %d deleted", mdelete), mdelete);
  printf("\n");

  /* Set the cursor.  */
  set_cursor ((first_new == 0) ? ((first_unread == 0) ?
				    1 : first_unread) : first_new) ;
  return 0;
}
