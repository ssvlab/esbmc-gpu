/* GNU mailutils - a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2007, 2009, 2010 Free Software Foundation,
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

#include "imap4d.h"

/*
6.3.6.  SUBSCRIBE Command

   Arguments:  mailbox

   Responses:  no specific responses for this command

   Result:     OK - subscribe completed
               NO - subscribe failure: can't subscribe to that name
               BAD - command unknown or arguments invalid
*/
int
imap4d_subscribe (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *name;
  char *file;
  FILE *fp;

  if (imap4d_tokbuf_argc (tok) != 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  name = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);

  asprintf (&file, "%s/.mailboxlist", real_homedir);
  fp = fopen (file, "a");
  free (file);
  if (fp)
    {
      fputs (name, fp);
      fputs ("\n", fp);
      fclose (fp);
      return util_finish (command, RESP_OK, "Completed");
    }
  return util_finish (command, RESP_NO, "Cannot subscribe");
}
