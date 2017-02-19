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
6.3.9.  LSUB Command

   Arguments:  reference name
               mailbox name with possible wildcards

   Responses:  untagged responses: LSUB

   Result:     OK - lsub completed
               NO - lsub failure: can't list that reference or name
               BAD - command unknown or arguments invalid
*/
int
imap4d_lsub (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *ref;
  char *wcard;
  char *file = NULL;
  char *pattern = NULL;
  const char *delim = "/";
  FILE *fp;
  
  if (imap4d_tokbuf_argc (tok) != 4)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  ref = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  wcard = imap4d_tokbuf_getarg (tok, IMAP4_ARG_2);

  asprintf (&pattern, "%s%s", ref, wcard);
  if (!pattern)
    return util_finish (command, RESP_NO, "Not enough memory");
  
  asprintf (&file, "%s/.mailboxlist", real_homedir);
  if (!file)
    {
      free (pattern);
      return util_finish (command, RESP_NO, "Not enough memory");
    }
  
  fp = fopen (file, "r");
  free (file);
  if (fp)
    {
      char *buf = NULL;
      size_t n = 0;
	
      while (getline (&buf, &n, fp) > 0)
	{
	  int len = strlen (buf);
	  if (buf[len - 1] == '\n')
	    buf[len - 1] = '\0';
	  if (util_wcard_match (buf, pattern, delim) == 0)
	    util_out (RESP_NONE, "LIST () \"%s\" %s", delim, buf);
	}
      fclose (fp);
      free (buf);
      return util_finish (command, RESP_OK, "Completed");
    }
  else if (errno == ENOENT)
    return util_finish (command, RESP_OK, "Completed");
  return util_finish (command, RESP_NO, "Cannot list subscriber");
}
