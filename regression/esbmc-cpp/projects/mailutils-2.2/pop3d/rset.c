/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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

/* Resets the connection so that no messages are marked as deleted */

int
pop3d_rset (char *arg)
{
  size_t i;
  size_t total = 0;

  if (strlen (arg) != 0)
    return ERR_BAD_ARGS;

  if (state != TRANSACTION)
    return ERR_WRONG_STATE;

  mu_mailbox_messages_count (mbox, &total);

  for (i = 1; i <= total; i++)
    {
      mu_message_t msg = NULL;
      mu_attribute_t attr = NULL;
      mu_mailbox_get_message (mbox, i, &msg);
      mu_message_get_attribute (msg, &attr);
      pop3d_unset_deleted (attr);
    }
  pop3d_outf ("+OK\r\n");
  return OK;
}
