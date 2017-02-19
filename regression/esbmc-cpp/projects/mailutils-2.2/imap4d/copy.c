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
6.4.7.  COPY Command

   Arguments:  message set
               mailbox name

   Responses:  no specific responses for this command

   Result:     OK - copy completed
               NO - copy error: can't copy those messages or to that
                    name
               BAD - command unknown or arguments invalid

  copy messages in argv[2] to mailbox in argv[3]
 */

int
imap4d_copy (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int rc;
  char *text;

  if (imap4d_tokbuf_argc (tok) != 4)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  rc = imap4d_copy0 (tok, 0, &text);
  
  if (rc == RESP_NONE)
    {
      /* Reset the state ourself.  */
      int new_state = (rc == RESP_OK) ? command->success : command->failure;
      if (new_state != STATE_NONE)
	state = new_state;
      return util_send ("%s %s\r\n", command->tag, text);
    }
  return util_finish (command, rc, "%s", text);
}

int
imap4d_copy0 (imap4d_tokbuf_t tok, int isuid, char **err_text)
{
  int status;
  char *msgset;
  char *name;
  char *mailbox_name;
  const char *delim = "/";
  size_t *set = NULL;
  int n = 0;
  mu_mailbox_t cmbox = NULL;
  int arg = IMAP4_ARG_1 + !!isuid;
  int ns;
  
  if (imap4d_tokbuf_argc (tok) != arg + 2)
    {
      *err_text = "Invalid arguments";
      return 1;
    }
  
  msgset = imap4d_tokbuf_getarg (tok, arg);
  name = imap4d_tokbuf_getarg (tok, arg + 1);
  /* Get the message numbers in set[].  */
  status = util_msgset (msgset, &set, &n, isuid);
  if (status != 0)
    {
      /* See RFC 3501, section 6.4.8, and a comment to the equivalent code
	 in fetch.c */
      *err_text = "Completed";
      return RESP_OK;
    }

  mailbox_name = namespace_getfullpath (name, delim, &ns);

  if (!mailbox_name)
    {
      *err_text = "NO Copy failed.";
      return RESP_NO;
    }

  /* If the destination mailbox does not exist, a server should return
     an error.  */
  status = mu_mailbox_create_default (&cmbox, mailbox_name);
  if (status == 0)
    {
      /* It SHOULD NOT automatifcllly create the mailbox. */
      status = mu_mailbox_open (cmbox, MU_STREAM_RDWR | mailbox_mode[ns]);
      if (status == 0)
	{
	  size_t i;
	  for (i = 0; i < n; i++)
	    {
	      mu_message_t msg = NULL;
	      size_t msgno = (isuid) ? uid_to_msgno (set[i]) : set[i];
	      if (msgno && mu_mailbox_get_message (mbox, msgno, &msg) == 0)
		mu_mailbox_append_message (cmbox, msg);
	    }
	  mu_mailbox_close (cmbox);
	}
      mu_mailbox_destroy (&cmbox);
    }
  free (set);
  free (mailbox_name);

  if (status == 0)
    {
      *err_text = "Completed";
      return RESP_OK;
    }

  /* Unless it is certain that the destination mailbox cannot be created,
     the server MUST send the response code "[TRYCREATE]" as the prefix
     of the text of the tagged NO response.  This gives a hint to the
     client that it can attempt a CREATE command and retry the copy if
     the CREATE is successful.  */
  *err_text = "[TRYCREATE] failed";
  return RESP_NO;
}
