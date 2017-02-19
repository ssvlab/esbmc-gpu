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
 * w[rite] [file] -- GNU extension
 * w[rite] [msglist] file
 * W[rite] [msglist] -- GNU extension
 */

int
mail_write (int argc, char **argv)
{
  mu_message_t msg;
  mu_body_t bod;
  mu_stream_t stream;
  FILE *output;
  char *filename = NULL;
  msgset_t *msglist = NULL, *mp;
  int sender = 0;
  size_t total_size = 0, total_lines = 0, size;

  if (mu_isupper (argv[0][0]))
    sender = 1;
  else if (argc >= 2)
    filename = util_outfolder_name (argv[--argc]);
  else
    {
      size_t n = get_cursor ();
      char *p = NULL;
      if (n == 0)
        {
          util_error (_("No applicable message"));
          return 1;
        }
      asprintf (&p, "%lu", (unsigned long) n);
      filename = util_outfolder_name (p);
      free (p);
    }
		
  if (msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &msglist))
    {
      if (filename)
          free (filename);
      return 1;
    }

  if (sender)
    {
      filename = util_outfolder_name (util_get_sender(msglist->msg_part[0], 1));
      if (!filename)
	{
	  msgset_free (msglist);
	  return 1;
	}
    }

  output = fopen (filename, "a");
  if (!output)
    {
      util_error (_("can't open %s: %s"), filename, strerror (errno));
      free (filename);
      fclose (output);
      msgset_free (msglist);
      return 1;
    }

  for (mp = msglist; mp; mp = mp->next)
    {
      mu_attribute_t attr;
      char buffer[512];
      off_t off = 0;
      size_t n = 0;

      if (util_get_message (mbox, mp->msg_part[0], &msg))
        continue;

      mu_message_get_body (msg, &bod);

      mu_body_size (bod, &size);
      total_size += size;
      mu_body_lines (bod, &size);
      total_lines += size;

      mu_body_get_stream (bod, &stream);
      /* should there be a separator? */
      while (mu_stream_read(stream, buffer, sizeof (buffer) - 1, off, &n) == 0
	     && n != 0)
	{
	  buffer[n] = '\0';
	  fprintf (output, "%s", buffer);
	  off += n;
	}

      /* mark as saved. */

      mu_message_get_attribute (msg, &attr);
      mu_attribute_set_userflag (attr, MAIL_ATTRIBUTE_SAVED);
    }

  fprintf (ofile, "\"%s\" %3lu/%-5lu\n", filename,
	   (unsigned long) total_lines, (unsigned long) total_size);

  free (filename);
  fclose (output);
  msgset_free (msglist);
  return 0;
}
