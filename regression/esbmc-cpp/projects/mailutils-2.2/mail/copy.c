/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2005, 2007, 2009, 2010 Free
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

#include "mail.h"

/*
 * c[opy] [file]
 * c[opy] [msglist] file
 * C[opy] [msglist]
 */

/*
 * mail_copy0() is shared between mail_copy() and mail_save().
 * argc, argv -- argument count & vector
 * mark -- whether we should mark the message as saved.
 */
int
mail_copy0 (int argc, char **argv, int mark)
{
  mu_message_t msg;
  mu_mailbox_t mbx;
  char *filename = NULL;
  msgset_t *msglist = NULL, *mp;
  int sender = 0;
  size_t total_size = 0, total_lines = 0, size;
  int status;

  if (mu_isupper (argv[0][0]))
    sender = 1;
  else if (argc >= 2)
    filename = mail_expand_name (argv[--argc]);
  else
    filename = strdup ("mbox");

  if (msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &msglist))
    {
      if (filename)
	free (filename);
      return 1;
    }

  if (sender)
    filename = util_outfolder_name (util_get_sender (msglist->msg_part[0], 1));

  if (!filename)
    {
      msgset_free (msglist);
      return 1;
    }

  if ((status = mu_mailbox_create_default (&mbx, filename)) != 0)
    {
      util_error (_("Cannot create mailbox %s: %s"), filename, 
                   mu_strerror (status));
      free (filename);
      msgset_free (msglist);
      return 1;
    }
  if ((status = mu_mailbox_open (mbx, MU_STREAM_WRITE | MU_STREAM_CREAT)) != 0)
    {
      util_error (_("Cannot open mailbox %s: %s"), filename, 
                   mu_strerror (status));
      free (filename);
      msgset_free (msglist);
      return 1;
    }

  for (mp = msglist; mp; mp = mp->next)
    {
      status = util_get_message (mbox, mp->msg_part[0], &msg);
      if (status)
        break;

      status = mu_mailbox_append_message (mbx, msg);
      if (status)
	{
	  util_error (_("Cannot append message: %s"), mu_strerror (status));
	  break;
	}
      
      mu_message_size (msg, &size);
      total_size += size;
      mu_message_lines (msg, &size);
      total_lines += size;

      if (mark)
 	{
	  mu_attribute_t attr;
	  mu_message_get_attribute (msg, &attr);
	  mu_attribute_set_userflag (attr, MAIL_ATTRIBUTE_SAVED);
	}
    }

  if (status == 0)
    fprintf (ofile, "\"%s\" %3lu/%-5lu\n", filename,
	     (unsigned long) total_lines, (unsigned long) total_size);

  mu_mailbox_close (mbx);
  mu_mailbox_destroy (&mbx);

  free (filename);
  msgset_free (msglist);
  return 0;
}

int
mail_copy (int argc, char **argv)
{
  return mail_copy0 (argc, argv, 0);
}
