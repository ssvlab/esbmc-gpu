/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

/* Enters the UPDATE phase and deletes marked messages */
/* Note:
   Whether the removal was successful or not, the server
   then releases any exclusive-access lock on the maildrop
   and closes the TCP connection.  */

static void pop3d_fix_mark ();

int
pop3d_quit (char *arg)
{
  int err = OK;
  if (strlen (arg) != 0)
    return ERR_BAD_ARGS;

  if (state == TRANSACTION)
    {
      pop3d_unlock ();
      pop3d_fix_mark ();

      if (mu_mailbox_flush (mbox, 1) != 0)
	err = ERR_FILE;
      if (mu_mailbox_close (mbox) != 0) 
	err = ERR_FILE;
      mu_mailbox_destroy (&mbox);
      mu_diag_output (MU_DIAG_INFO, _("session ended for user: %s"), username);
    }
  else
    mu_diag_output (MU_DIAG_INFO, _("session ended for no user"));

  state = UPDATE;
  update_login_delay (username);
  free (username);
  free (md5shared);

  if (err == OK)
    pop3d_outf ("+OK\r\n");
  return err;
}


static void
pop3d_fix_mark ()
{
  size_t i;
  size_t total = 0;
  char *value = NULL;
  
  mu_mailbox_messages_count (mbox, &total);

  for (i = 1; i <= total; i++)
    {
      mu_message_t msg = NULL;
      mu_attribute_t attr = NULL;
       
      mu_mailbox_get_message (mbox, i, &msg);
      mu_message_get_attribute (msg, &attr);
      
      if (pop3d_is_deleted (attr))
	mu_attribute_set_deleted (attr);

      expire_mark_message (msg, &value);
    }
  
  free (value);
}
