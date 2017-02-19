/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
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

#include "guimb.h"

char *temp_filename;
FILE *temp_file;
mu_mailbox_t mbox;

void
collect_open_default ()
{
  size_t nmesg;

  if (!default_mailbox)
    {
      int rc = mu_construct_user_mailbox_url (&default_mailbox, user_name);
      if (rc)
	{
	  util_error (_("cannot construct default mailbox URL: %s"),
                      mu_strerror (rc));
	  exit (1);
	}
    }
  if (mu_mailbox_create (&mbox, default_mailbox) != 0
      || mu_mailbox_open (mbox, MU_STREAM_RDWR) != 0)
    {
      util_error (_("cannot open default mailbox %s: %s"),
		  default_mailbox, mu_strerror (errno));
      exit (1);
    }

  /* Suck in the messages */
  mu_mailbox_messages_count (mbox, &nmesg);
}

/* Open temporary file for collecting incoming messages */
void
collect_open_mailbox_file ()
{
  int fd;

  /* Create input mailbox */
  fd = mu_tempfile (NULL, &temp_filename);
  if (fd == -1)
    exit (1);

  temp_file = fdopen (fd, "w");
  if (!temp_file)
    {
      util_error ("fdopen: %s", strerror (errno));
      close (fd);
      exit (1);
    }
}

/* Append contents of file `name' to the temporary file */
int
collect_append_file (char *name)
{
  char *buf = NULL;
  size_t n = 0;
  FILE *fp;

  if (strcmp (name, "-") == 0)
    fp = stdin;
  else
    {
      fp = fopen (name, "r");
      if (!fp)
	{
	  util_error (_("cannot open input file %s: %s"), name, strerror (errno));
	  return -1;
	}
    }

  /* Copy the contents of the file */
  while (getline (&buf, &n, fp) > 0)
    fprintf (temp_file, "%s", buf);

  free (buf);
  fclose (fp);
  return 0;
}

/* Close the temporary, and reopen it as a mailbox. */
void
collect_create_mailbox ()
{
  size_t nmesg;

  if (!temp_file)
    return;
  
  fclose (temp_file);

  if (mu_mailbox_create (&mbox, temp_filename) != 0
      || mu_mailbox_open (mbox, MU_STREAM_READ) != 0)
    {
      util_error (_("cannot create temp mailbox %s: %s"),
		  temp_filename, strerror (errno));
      unlink (temp_filename);
      exit (1);
    }

  /* Suck in the messages */
  mu_mailbox_messages_count (mbox, &nmesg);

  if (nmesg == 0)
    {
      util_error (_("input format not recognized"));
      exit (1);
    }
}

int
collect_output ()
{
  size_t i, count = 0;
  mu_mailbox_t outbox = NULL;
  int saved_umask;

  if (!temp_filename)
    {
      mu_mailbox_expunge (mbox);
      return 0;
    }

  if (user_name)
    saved_umask = umask (077);
  
  if (mu_mailbox_create_default (&outbox, default_mailbox) != 0
      || mu_mailbox_open (outbox, MU_STREAM_RDWR|MU_STREAM_CREAT) != 0)
    {
      mu_mailbox_destroy (&outbox);
      mu_error (_("cannot open output mailbox %s: %s"),
		default_mailbox, strerror (errno));
      return 1;
    }

  mu_mailbox_messages_count (mbox, &count);
  for (i = 1; i <= count; i++)
    {
      mu_message_t msg = NULL;
      mu_attribute_t attr = NULL;

      mu_mailbox_get_message (mbox, i, &msg);
      mu_message_get_attribute (msg, &attr);
      if (!mu_attribute_is_deleted (attr))
	{
	  mu_attribute_set_recent (attr);
	  mu_mailbox_append_message (outbox, msg);
	}
    }

  mu_mailbox_close (outbox);
  mu_mailbox_destroy (&outbox);

  if (user_name)
    umask (saved_umask);
  return 0;
}

  
/* Close the temporary mailbox and unlink the file associated with it */
void
collect_drop_mailbox ()
{
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  if (temp_filename)
    {
      unlink (temp_filename);
      free (temp_filename);
    }
}

