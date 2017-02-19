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
6.3.4.  DELETE Command

   Arguments:  mailbox name

   Responses:  no specific responses for this command

   Result:     OK - delete completed
               NO - delete failure: can't delete mailbox with that name
               BAD - command unknown or arguments invalid
*/  
int
imap4d_delete (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int rc = RESP_OK;
  const char *msg = "Completed";
  const char *delim = "/";
  char *name;

  if (imap4d_tokbuf_argc (tok) != 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  name = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  if (!name || *name == '\0')
    return util_finish (command, RESP_BAD, "Too few arguments");

  /* It is an error to attempt to delele "INBOX or a mailbox
     name that dos not exists.  */
  if (mu_c_strcasecmp (name, "INBOX") == 0)
    return util_finish (command, RESP_NO, "Already exist");

 /* Allocates memory.  */
  name = namespace_getfullpath (name, delim, NULL);
  if (!name)
    return util_finish (command, RESP_NO, "Cannot remove");

  if (remove (name) != 0)
    {
      rc = RESP_NO;
      msg = "Cannot remove";
    }
  return util_finish (command, rc, msg);
}
