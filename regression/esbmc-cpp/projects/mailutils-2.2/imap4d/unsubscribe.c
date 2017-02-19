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

struct scan_data
{
  int result;
  char *name;
  FILE *tmp;
};

static int
scan_mailbox_list (char *filename,
		   int (*handler) (struct scan_data *data, char *name),
		   struct scan_data *data)
{
  FILE *fp;
  char buffer[124];

  fp = fopen (filename, "r");
  if (!fp)
    return -1;
    
  while (fgets (buffer, sizeof (buffer), fp))
    {
      size_t n = strlen (buffer);
      if (n && buffer[n - 1] == '\n')
	buffer[n - 1] = '\0';
      if (handler (data, buffer))
	break;
    }
  fclose (fp);
  return 0;
}

static int
scan_only (struct scan_data *data, char *name)
{
  if (strcmp (data->name, name) == 0)
    {
      data->result = 1;
      return 1;
    }
  return 0;
}

static int
unsubscribe (struct scan_data *data, char *name)
{
  if (strcmp (data->name, name))
    {
      fputs (name, data->tmp);
      fputs ("\n", data->tmp);
    }
  return 0;
}

/*
6.3.7.  UNSUBSCRIBE Command

   Arguments:  mailbox name

   Responses:  no specific responses for this command

   Result:     OK - unsubscribe completed
               NO - unsubscribe failure: can't unsubscribe that name
               BAD - command unknown or arguments invalid

      The UNSUBSCRIBE command removes the specified mailbox name from
      the server's set of "active" or "subscribed" mailboxes as returned
      by the LSUB command.  This command returns a tagged OK response
      only if the unsubscription is successful.
*/
int
imap4d_unsubscribe (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *name;
  char *file;
  struct scan_data sd;
  int rc;
  
  if (imap4d_tokbuf_argc (tok) != 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  name = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);

  asprintf (&file, "%s/.mailboxlist", real_homedir);
  sd.result = 0;
  sd.name = name;

  rc = scan_mailbox_list (file, scan_only, &sd); 
  if (rc == 0)
    {
      if (sd.result)
	{
	  char *tmpname = NULL;
	  asprintf (&tmpname, "%s.%d", file, getpid ());
	  sd.tmp = fopen (tmpname, "a");
	  if (!sd.tmp)
	    rc = -1;
	  else
	    {
	      rc = scan_mailbox_list (file, unsubscribe, &sd);
	      fclose (sd.tmp);
	      if (rc == 0)
		rename (tmpname, file);
	    }
	  free (tmpname);
	}
    }

  free (file);
  if (rc)
    return util_finish (command, RESP_NO, "Cannot unsubscribe");

  return util_finish (command, RESP_OK, "Completed");
}
