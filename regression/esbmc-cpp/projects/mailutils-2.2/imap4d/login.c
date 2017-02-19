/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2006, 2007, 2008, 2009, 2010 Free
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
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "imap4d.h"

/*
6.2.2.  LOGIN Command

   Arguments:  user name
               password

   Responses:  no specific responses for this command

   Result:     OK - login completed, now in authenticated state
               NO - login failure: user name or password rejected
               BAD - command unknown or arguments invalid
*/  
int
imap4d_login (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *username, *pass;
  int rc;

  if (login_disabled || tls_required)    
    return util_finish (command, RESP_NO, "Command disabled");

  if (imap4d_tokbuf_argc (tok) != 4)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  username = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  pass = imap4d_tokbuf_getarg (tok, IMAP4_ARG_2);

  auth_data = mu_get_auth_by_name (username);

  if (auth_data == NULL)
    {
      mu_diag_output (MU_DIAG_INFO, _("user `%s' nonexistent"), username);
      return util_finish (command, RESP_NO, "User name or passwd rejected");
    }

  rc = mu_authenticate (auth_data, pass);
  openlog (MU_LOG_TAG (), LOG_PID, mu_log_facility);
  if (rc)
    {
      mu_diag_output (MU_DIAG_INFO, _("login failed: %s"), username);
      return util_finish (command, RESP_NO, "User name or passwd rejected");
    }

  if (imap4d_session_setup0 ())
    return util_finish (command, RESP_NO, "User name or passwd rejected");
  return util_finish (command, RESP_OK, "Completed");
}

