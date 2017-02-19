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
#include <unistd.h>

#define MKDIR_PERMISSIONS 0700

/* FIXME: take permissions as argument */
static int
mkdir_p (char *name, int delim)
{
  char *p, *dir;
  int rc = 0;

  dir = name;
  if (dir[0] == delim)
    dir++;
  for (; rc == 0 && (p = strchr (dir, delim)); dir = p)
    {
      struct stat st;
      
      *p++ = 0;
      if (stat (name, &st) == 0)
	{
	  if (!S_ISDIR (st.st_mode))
	    {
	      mu_diag_output (MU_DIAG_ERR,
			      _("component %s is not a directory"), name);
	      rc = 1;
	    }
	}
      else if (errno != ENOENT)
	{
	  mu_diag_output (MU_DIAG_ERR,
			  _("cannot stat file %s: %s"),
			  name, mu_strerror (errno));
	  rc = 1;
	}
      else if (mkdir (name, MKDIR_PERMISSIONS))
	{
	  mu_diag_output (MU_DIAG_ERR,
			  _("cannot create directory %s: %s"), name,
			  mu_strerror (errno));
	  rc = 1;
	}
      p[-1] = delim;
    }
  return 0;
}

/*
6.3.3.  CREATE Command

   Arguments:  mailbox name

   Responses:  no specific responses for this command

   Result:     OK - create completed
               NO - create failure: can't create mailbox with that name
               BAD - command unknown or arguments invalid
*/  
/* FIXME: How do we do this ??????:
   IF a new mailbox is created with the same name as a mailbox which was
   deleted, its unique identifiers MUST be greater than any unique identifiers
   used in the previous incarnation of the mailbox.  */
int
imap4d_create (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *name;
  const char *delim = "/";
  int isdir = 0;
  int ns;
  int rc = RESP_OK;
  const char *msg = "Completed";

  if (imap4d_tokbuf_argc (tok) != 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  name = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);

  if (*name == '\0')
    return util_finish (command, RESP_BAD, "Too few arguments");

  /* Creating, "Inbox" should always fail.  */
  if (mu_c_strcasecmp (name, "INBOX") == 0)
    return util_finish (command, RESP_BAD, "Already exist");

  /* RFC 3501:
         If the mailbox name is suffixed with the server's hierarchy
	 separator character, this is a declaration that the client intends
	 to create mailbox names under this name in the hierarchy.

     The trailing delimiter will be removed by namespace normalizer, so
     test for it now.
  */
  if (name[strlen (name) - 1] == delim[0])
    isdir = 1;
  
  /* Allocates memory.  */
  name = namespace_getfullpath (name, delim, &ns);

  if (!name)
    return util_finish (command, RESP_NO, "Cannot create mailbox");

  /* It will fail if the mailbox already exists.  */
  if (access (name, F_OK) != 0)
    {
      if (mkdir_p (name, delim[0]))
	{
	  rc = RESP_NO;
	  msg = "Cannot create mailbox";
	}
      
      if (rc == RESP_OK && !isdir)
	{
	  mu_mailbox_t mbox;
	  
	  rc = mu_mailbox_create_default (&mbox, name);
	  if (rc)
	    {
	      mu_diag_output (MU_DIAG_ERR,
			      _("Cannot create mailbox %s: %s"), name,
			      mu_strerror (rc));
	      rc = RESP_NO;
	      msg = "Cannot create mailbox";
	    }
	  else if ((rc = mu_mailbox_open (mbox,
					  MU_STREAM_RDWR | MU_STREAM_CREAT
					  | mailbox_mode[ns])))
	    {
	      mu_diag_output (MU_DIAG_ERR,
			      _("Cannot open mailbox %s: %s"),
			      name, mu_strerror (rc));
	      rc = RESP_NO;
	      msg = "Cannot create mailbox";
	    }
	  else
	    {
	      mu_mailbox_close (mbox);
	      mu_mailbox_destroy (&mbox);
	      rc = RESP_OK;
	    }
	}
    }
  else
    {
      rc = RESP_NO;
      msg = "already exists";
    }

  return util_finish (command, rc, msg);
}
