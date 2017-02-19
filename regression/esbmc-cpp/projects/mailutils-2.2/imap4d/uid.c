/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2007, 2008, 2009, 2010 Free Software
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

/*
 * see comment in store.c, raise to the nth power
 * this is really going to sux0r (maybe)
 */

int
imap4d_uid (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *cmd;
  int rc = RESP_NO;
  char *err_text = "Completed";

  if (imap4d_tokbuf_argc (tok) < 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  cmd = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  
  if (mu_c_strcasecmp (cmd, "FETCH") == 0)
    rc = imap4d_fetch0 (tok, 1, &err_text);
  else if (mu_c_strcasecmp (cmd, "COPY") == 0)
    rc = imap4d_copy0 (tok, 1, &err_text);
  else if (mu_c_strcasecmp (cmd, "STORE") == 0)
    rc = imap4d_store0 (tok, 1, &err_text);
  else if (mu_c_strcasecmp (cmd, "SEARCH") == 0)
    rc = imap4d_search0 (tok, 1, &err_text);
  else
    {
      err_text = "Uknown uid command";
      rc = RESP_BAD;
    }
  return util_finish (command, rc, "%s %s", cmd, err_text);
}
