/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2004, 2005, 2007, 2008, 2010 Free Software
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

#include "imap4d.h"

static int
imap4d_close0 (struct imap4d_command *command, imap4d_tokbuf_t tok,
	       int expunge)
{
  const char *msg = NULL;
  int status, flags;

  if (imap4d_tokbuf_argc (tok) != 2)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  mu_mailbox_get_flags (mbox, &flags);
  if ((flags & MU_STREAM_READ) == 0)
    {
      status = mu_mailbox_flush (mbox, expunge);
      if (status)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "mu_mailbox_flush", NULL, status);
	  msg = "flushing mailbox failed";
	}
    }
  
  /* No messages are removed, and no error is given, if the mailbox is
     selected by an EXAMINE command or is otherwise selected read-only.  */
  status = mu_mailbox_close (mbox);
  if (status)
    {
      mu_diag_funcall (MU_DIAG_ERROR, "mu_mailbox_close", NULL, status);
      msg = "closing mailbox failed";
    }
  mu_mailbox_destroy (&mbox);

  if (msg)
    return util_finish (command, RESP_NO, msg);
  return util_finish (command, RESP_OK, "Completed");
}

/*
6.4.2.  CLOSE Command

   Arguments:  none

   Responses:  no specific responses for this command

   Result:     OK - close completed, now in authenticated state
               NO - close failure: no mailbox selected
               BAD - command unknown or arguments invalid

   The CLOSE command permanently removes from the currently selected
   mailbox all messages that have the \\Deleted flag set, and returns
   to authenticated state from selected state.  */
int
imap4d_close (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  return imap4d_close0 (command, tok, 1);
}

/* RFC 3691:
   
2.  UNSELECT Command

   Arguments:  none

   Responses:  no specific responses for this command

   Result:     OK - unselect completed, now in authenticated state
               BAD - no mailbox selected, or argument supplied but
                     none permitted

      The UNSELECT command frees server's resources associated with the
      selected mailbox and returns the server to the authenticated
      state.  This command performs the same actions as CLOSE, except
      that no messages are permanently removed from the currently
      selected mailbox.
*/
int
imap4d_unselect (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  return imap4d_close0 (command, tok, 0);
}
