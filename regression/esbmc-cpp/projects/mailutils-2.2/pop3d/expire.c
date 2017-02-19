/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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

#include "pop3d.h"

/* EXPIRE see RFC2449:

  Implementation:
    When a message is downloaded by RETR or TOP, it is marked with
    "X-Expire-Timestamp: N", where N is the current value of
    UNIX timestamp.

    If pop3d was started with --delete-expired, the messages whose
    X-Expire-Timestamp is more than (time(NULL)-expire days) old
    are deleted.

    Otherwise, such messages remain in the mailbox and the system
    administrator is supposed to run a cron job that purges the mailboxes
    (easily done using GNU sieve timestamp extension).

*/

void
pop3d_mark_retr (mu_attribute_t attr)
{
  mu_attribute_set_userflag (attr, POP3_ATTRIBUTE_RETR);
}
 
int
pop3d_is_retr (mu_attribute_t attr)
{
  return mu_attribute_is_userflag (attr, POP3_ATTRIBUTE_RETR);
}
 
void
pop3d_unmark_retr (mu_attribute_t attr)
{
  if (mu_attribute_is_userflag (attr, POP3_ATTRIBUTE_RETR))
    mu_attribute_unset_userflag (attr, POP3_ATTRIBUTE_RETR);
}

static int
header_is_expired (mu_header_t hdr)
{
  time_t timestamp;
  char buf[64];
  char *p;
  
  if (!expire_on_exit)
    return 0;
  if (mu_header_get_value (hdr, MU_HEADER_X_EXPIRE_TIMESTAMP,
			buf, sizeof buf, NULL))
    return 0;
  timestamp = strtoul (buf, &p, 0);
  while (*p && mu_isspace (*p))
    p++;
  if (*p)
    return 0;
  return time (NULL) >= timestamp + expire * 86400;
}

/* If pop3d is started with --expire, add an expiration header to the message.
   Additionally, if --deltete-expired option was given, mark
   the message as deleted if its X-Expire-Timestamp is too old.
   Arguments:
      msg   - Message to operate upon
      value - Points to a character buffer where the value of
              X-Expire-Timestamp is to be stored. *value must be set to
	      NULL upon the first invocation of this function */
void
expire_mark_message (mu_message_t msg, char **value)
{
  /* Mark the message with a timestamp. */
  if (expire != EXPIRE_NEVER)
    {
      mu_header_t header = NULL;
      mu_attribute_t attr = NULL;
      
      if (!*value)
	asprintf (value, "%lu", (unsigned long) time (NULL));
      
      mu_message_get_header (msg, &header);
      mu_message_get_attribute (msg, &attr);
	  
      if (pop3d_is_retr (attr))
	mu_header_set_value (header, MU_HEADER_X_EXPIRE_TIMESTAMP, *value, 0);
      
      if (header_is_expired (header))
	mu_attribute_set_deleted (attr);
    }
}
