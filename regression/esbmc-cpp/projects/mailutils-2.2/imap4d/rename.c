/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2008, 2009, 2010 Free Software
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
6.3.5.  RENAME Command

   Arguments:  existing mailbox name
               new mailbox name

   Responses:  no specific responses for this command

   Result:     OK - rename completed
               NO - rename failure: can't rename mailbox with that name,
                    can't rename to mailbox with that name
               BAD - command unknown or arguments invalid
*/  
/*
  FIXME: Renaming a mailbox we must change the UIDVALIDITY
  of the mailbox.  */

int
imap4d_rename (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *oldname;
  char *newname;
  int rc = RESP_OK;
  const char *msg = "Completed";
  struct stat newst;
  const char *delim = "/";
  int ns;
  
  if (imap4d_tokbuf_argc (tok) != 4)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  oldname = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  newname = imap4d_tokbuf_getarg (tok, IMAP4_ARG_2);

  if (mu_c_strcasecmp (newname, "INBOX") == 0)
    return util_finish (command, RESP_NO, "Name Inbox is reservered");

  /* Allocates memory.  */
  newname = namespace_getfullpath (newname, delim, &ns);
  if (!newname)
    return util_finish (command, RESP_NO, "Permission denied");

  /* It is an error to attempt to rename from a mailbox name that already
     exist.  */
  if (stat (newname, &newst) == 0)
    {
      if (!S_ISDIR(newst.st_mode))
	{
	  free (newname);
	  return util_finish (command, RESP_NO, "Already exist, delete first");
	}
    }

  /* Renaming INBOX is permitted, and has special behavior.  It moves
     all messages in INBOX to a new mailbox with the given name,
     leaving INBOX empty.  */
  if (mu_c_strcasecmp (oldname, "INBOX") == 0)
    {
      mu_mailbox_t newmbox = NULL;
      mu_mailbox_t inbox = NULL;

      if (S_ISDIR (newst.st_mode))
	{
	  free (newname);
	  return util_finish (command, RESP_NO, "Cannot be a directory");
	}
      if (mu_mailbox_create (&newmbox, newname) != 0
	  || mu_mailbox_open (newmbox,
			      MU_STREAM_CREAT | MU_STREAM_RDWR
			        | mailbox_mode[ns]) != 0)
	{
	  free (newname);
	  return util_finish (command, RESP_NO, "Cannot create new mailbox");
	}
      free (newname);

      if (mu_mailbox_create_default (&inbox, auth_data->name) == 0 &&
	  mu_mailbox_open (inbox, MU_STREAM_RDWR) == 0)
	{
	  size_t no;
	  size_t total = 0;
	  mu_mailbox_messages_count (inbox, &total);
	  for (no = 1; no <= total; no++)
	    {
	      mu_message_t message;
	      if (mu_mailbox_get_message (inbox, no, &message) == 0)
		{
		  mu_attribute_t attr = NULL;
		  mu_mailbox_append_message (newmbox, message);
		  mu_message_get_attribute (message, &attr);
		  mu_attribute_set_deleted (attr);
		}
	    }
	  mu_mailbox_expunge (inbox);
	  mu_mailbox_close (inbox);
	  mu_mailbox_destroy (&inbox);
	}
      mu_mailbox_close (newmbox);
      mu_mailbox_destroy (&newmbox);
      return util_finish (command, RESP_OK, "Already exist");
    }

  oldname = namespace_getfullpath (oldname, delim, NULL);

  /* It must exist.  */
  /* FIXME: 1. What if odlname or newname is a remote mailbox?
            2. If newname is local and is in another namespace, its
  	       permissions must be fixed */
  if (!oldname || rename (oldname, newname) != 0)
    {
      rc = RESP_NO;
      msg = "Failed";
    }
  if (oldname)
    free (oldname);
  free (newname);
  return util_finish (command, rc, msg);
}
