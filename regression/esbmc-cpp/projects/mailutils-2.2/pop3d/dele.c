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

/* DELE adds a message number to the list of messages to be deleted on QUIT */

int
pop3d_dele (char *arg)
{
  size_t num;
  mu_message_t msg;
  mu_attribute_t attr = NULL;

  if ((arg == NULL) || (strchr (arg, ' ') != NULL))
    return ERR_BAD_ARGS;

  if (state != TRANSACTION)
    return ERR_WRONG_STATE;

  num = strtoul (arg, NULL, 10);

  if (mu_mailbox_get_message (mbox, num, &msg) != 0)
    return ERR_NO_MESG;

  mu_message_get_attribute (msg, &attr);
  pop3d_mark_deleted (attr);
  pop3d_outf ("+OK Message %s marked\r\n", mu_umaxtostr (0, num));
  return OK;
}
