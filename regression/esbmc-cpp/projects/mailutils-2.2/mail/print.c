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

#include "mail.h"

/*
 * p[rint] [msglist]
 * t[ype] [msglist]
 * P[rint] [msglist]
 * T[ype] [msglist]
 */

static int
mail_print_msg (msgset_t *mspec, mu_message_t mesg, void *data)
{
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t stream;
  char buffer[512];
  off_t off = 0;
  size_t n = 0, lines = 0;
  FILE *out = ofile;
  int pagelines = util_get_crt ();
  
  mu_message_lines (mesg, &lines);
  if (mailvar_get (NULL, "showenvelope", mailvar_type_boolean, 0) == 0)
    lines++;
  
  /* If it is POP or IMAP the lines number is not known, so try
     to be smart about it.  */
  if (lines == 0)
    {
      if (pagelines)
	{
	  size_t col = (size_t)util_getcols ();
	  if (col)
	    {
	      size_t size = 0;
	      mu_message_size (mesg, &size);
	      lines =  size / col;
	    }
	}
    }

  if (pagelines && lines > pagelines)
    out = popen (getenv ("PAGER"), "w");

  if (mailvar_get (NULL, "showenvelope", mailvar_type_boolean, 0) == 0)
    print_envelope (mspec, mesg, "From");
  
  if (*(int *) data) /* print was called with a lowercase 'p' */
    {
      size_t i, num = 0;
      const char *sptr;
      char *tmp;
      
      mu_message_get_header (mesg, &hdr);
      mu_header_get_field_count (hdr, &num);

      for (i = 1; i <= num; i++)
	{
	  if (mu_header_sget_field_name (hdr, i, &sptr))
	    continue;
	  if (mail_header_is_visible (sptr))
	    {
	      fprintf (out, "%s: ", sptr);
	      mu_header_aget_field_value (hdr, i, &tmp);
	      if (mail_header_is_unfoldable (sptr))
		mu_string_unfold (tmp, NULL);
	      util_rfc2047_decode (&tmp);
	      fprintf (out, "%s\n", tmp);
	      free (tmp);
	    }
	}
      fprintf (out, "\n");
      mu_message_get_body (mesg, &body);
      mu_body_get_stream (body, &stream);
    }
  else
    mu_message_get_stream (mesg, &stream);
  
  while (mu_stream_read (stream, buffer, sizeof buffer - 1, off, &n) == 0
	 && n != 0)
    {
      if (ml_got_interrupt())
	{
	  util_error (_("\nInterrupt"));
	  break;
	}
      buffer[n] = '\0';
      fprintf (out, "%s", buffer);
      off += n;
    }
  if (out != ofile)
    pclose (out);
  
  util_mark_read (mesg);

  set_cursor (mspec->msg_part[0]);
  
  return 0;
}

int
mail_print (int argc, char **argv)
{
  int lower = mu_islower (argv[0][0]);
  int rc = util_foreach_msg (argc, argv, MSG_NODELETED|MSG_SILENT,
			     mail_print_msg, &lower);
  return rc;
}

