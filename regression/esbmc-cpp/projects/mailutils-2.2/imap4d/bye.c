/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

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

int
imap4d_bye (int reason)
{
  return imap4d_bye0 (reason, NULL);
}

int
imap4d_bye0 (int reason, struct imap4d_command *command)
{
  int status = EX_SOFTWARE;

  if (mbox)
    {
      mu_mailbox_flush (mbox, 0);
      mu_mailbox_close (mbox);
      mu_mailbox_destroy (&mbox);
    }

  switch (reason)
    {
    case ERR_NO_MEM:
      util_out (RESP_BYE, "Server terminating: no more resources.");
      mu_diag_output (MU_DIAG_ERROR, _("not enough memory"));
      break;

    case ERR_TERMINATE:
      status = EX_OK;
      util_out (RESP_BYE, "Server terminating on request.");
      mu_diag_output (MU_DIAG_NOTICE, _("terminating on request"));
      break;

    case ERR_SIGNAL:
      mu_diag_output (MU_DIAG_ERROR, _("quitting on signal"));
      exit (status);

    case ERR_TIMEOUT:
      status = EX_TEMPFAIL;
      util_out (RESP_BYE, "Session timed out");
      if (state == STATE_NONAUTH)
        mu_diag_output (MU_DIAG_INFO, _("session timed out for no user"));
      else
	mu_diag_output (MU_DIAG_INFO, _("session timed out for user: %s"), auth_data->name);
      break;

    case ERR_NO_OFILE:
      status = EX_IOERR;
      mu_diag_output (MU_DIAG_INFO, _("write error on control stream"));
      break;

    case ERR_NO_IFILE:
      status = EX_IOERR;
      mu_diag_output (MU_DIAG_INFO, _("read error on control stream"));
      break;

    case ERR_MAILBOX_CORRUPTED:
      status = EX_OSERR;
      mu_diag_output (MU_DIAG_ERROR, _("mailbox modified by third party"));
      break;
      
    case OK:
      status = EX_OK;
      util_out (RESP_BYE, "Session terminating.");
      if (state == STATE_NONAUTH)
	mu_diag_output (MU_DIAG_INFO, _("session terminating"));
      else
	mu_diag_output (MU_DIAG_INFO, _("session terminating for user: %s"), auth_data->name);
      break;

    default:
      util_out (RESP_BYE, "Quitting (reason unknown)");
      mu_diag_output (MU_DIAG_ERROR, _("quitting (numeric reason %d)"), reason);
      break;
    }

  if (status == EX_OK && command)
     util_finish (command, RESP_OK, "Completed");

  util_bye ();

  closelog ();
  exit (status);
}

