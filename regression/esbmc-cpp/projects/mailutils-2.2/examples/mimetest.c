/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include <sys/types.h>

#include <mailutils/mailutils.h>

void message_display_parts(mu_message_t msg, int indent);

const char *from;
const char *subject;
const char *charset;
int print_attachments;
int indent_level = 4;

void
print_file (const char *fname, int indent)
{
  char buf[128];
  FILE *fp = fopen (fname, "r");

  if (!fp)
    {
      fprintf (stderr, "can't open file %s: %s", fname, strerror (errno));
      return;
    }

  while (fgets (buf, sizeof buf, fp))
    printf ("%*.*s%s", indent, indent, "", buf);
  fclose (fp);
  unlink (fname);
}

int
main (int argc, char **argv)
{
  mu_mailbox_t mbox = NULL;
  size_t i;
  size_t count = 0;
  char *mailbox_name;
  int debug = 0;

  for (i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "-d") == 0)
        debug = 1;
      else if (strcmp (argv[i], "-p") == 0)
        print_attachments = 1;
      else if (strcmp (argv[i], "-i") == 0)
	{
	  if (++i == argc)
	    {
	      mu_error ("-i requires argument");
	      exit (1);
	    }
	  indent_level = strtoul (argv[i], NULL, 0);
	}
      else if (strcmp (argv[i], "-c") == 0)
	{
	  if (++i == argc)
	    {
	      mu_error ("-c requires argument");
	      exit (1);
	    }
	  charset = argv[i];
	}
      else
        break;
    }

  mailbox_name = argv[i];

  /* Registration.  */
  mu_registrar_record (mu_imap_record);
  mu_registrar_record (mu_pop_record);
  mu_registrar_record (mu_mbox_record);
  mu_registrar_set_default_record (mu_mbox_record);

  MU_ASSERT (mu_mailbox_create_default (&mbox, mailbox_name));
  
  /* Debugging trace. */
  if (debug)
    {
      mu_debug_t debug;
      mu_mailbox_get_debug (mbox, &debug);
      mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
    }

  /* Open the mailbox for reading only.  */
  MU_ASSERT (mu_mailbox_open (mbox, MU_STREAM_RDWR));

  /* Iterate through the entire message set.  */
  mu_mailbox_messages_count (mbox, &count);

  for (i = 1; i <= count; ++i)
    {
      mu_message_t msg;
      mu_header_t hdr;
      size_t nparts;
      size_t msize, nlines;

      MU_ASSERT (mu_mailbox_get_message (mbox, i, &msg));
      MU_ASSERT (mu_message_size (msg, &msize));
      MU_ASSERT (mu_message_lines (msg, &nlines));
      MU_ASSERT (mu_message_get_header (msg, &hdr));
      if (mu_header_sget_value (hdr, MU_HEADER_FROM, &from))
	from = "";
      if (mu_header_sget_value (hdr, MU_HEADER_SUBJECT, &subject))
	subject = "";
      printf ("Message: %lu\n", (unsigned long) i);
      printf ("From: %s\n", from);
      printf ("Subject: %s\n", subject);

      MU_ASSERT (mu_message_get_num_parts (msg, &nparts));
      printf ("Number of parts in message - %lu\n",
	      (unsigned long) nparts);
      printf ("Total message size - %lu/%lu\n",
	      (unsigned long) msize, (unsigned long) nlines);
      message_display_parts (msg, 0);
    }
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return 0;
}

char buf[2048];

static void
print_message_part_sizes (mu_message_t part, int indent)
{
  mu_body_t body;
  mu_header_t hdr;
  size_t msize, mlines, hsize, hlines, bsize, blines;
  
  MU_ASSERT (mu_message_size (part, &msize));
  MU_ASSERT (mu_message_lines (part, &mlines));
  MU_ASSERT (mu_message_get_header (part, &hdr));
  MU_ASSERT (mu_header_size (hdr, &hsize));
  MU_ASSERT (mu_header_lines (hdr, &hlines));
  MU_ASSERT (mu_message_get_body (part, &body));
  MU_ASSERT (mu_body_size (body, &bsize));
  MU_ASSERT (mu_body_lines (body, &blines));
  printf ("%*.*sMessage part size - %lu/%lu: %lu/%lu, %lu/%lu\n",
	  indent, indent, "",
	  (unsigned long) msize, (unsigned long) mlines,
	  (unsigned long) hsize, (unsigned long) hlines,
	  (unsigned long) bsize, (unsigned long) blines);
}
  
void
message_display_parts (mu_message_t msg, int indent)
{
  int ret, j;
  size_t nparts;
  mu_message_t part;
  mu_header_t hdr;
  mu_stream_t str;
  mu_body_t body;
  int offset, ismulti;
  size_t nbytes;

  /* How many parts does the message has? */
  if ((ret = mu_message_get_num_parts (msg, &nparts)) != 0)
    {
      fprintf (stderr, "mu_message_get_num_parts - %s\n", mu_strerror (ret));
      exit (2);
    }

  /* Iterate through all the parts.
     Treat type "message/rfc822" differently, since it is a message of
     its own that can have other subparts(recursive). */
  for (j = 1; j <= nparts; j++)
    {
      int status;
      const char *hvalue;
      char *type = NULL;
      const char *encoding = "";

      MU_ASSERT (mu_message_get_part (msg, j, &part));
      MU_ASSERT (mu_message_get_header (part, &hdr));
      status = mu_header_sget_value (hdr, MU_HEADER_CONTENT_TYPE,
				     &hvalue);
      if (status == MU_ERR_NOENT)
	/* nothing */;
      else if (status != 0)
	mu_error ("Cannot get header value: %s", mu_strerror (status));
      else
	{
	  status = mu_mimehdr_aget_disp (hvalue, &type);
	  if (status)
	    mu_error ("Cannot extract content type field: %s",
		      mu_strerror (status));
	}
      printf ("%*.*sType of part %d = %s\n", indent, indent, "",
	      j, type ? type : "");
      print_message_part_sizes (part, indent);
      if (mu_header_sget_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING,
				&encoding))
	encoding = "";
      ismulti = 0;
      if ((type
           && mu_c_strcasecmp (type, "message/rfc822") == 0)
          || (mu_message_is_multipart (part, &ismulti) == 0 && ismulti))
        {
          if (!ismulti)
	    MU_ASSERT (mu_message_unencapsulate (part, &part, NULL));
	  
          MU_ASSERT (mu_message_get_header (part, &hdr));
          if (mu_header_sget_value (hdr, MU_HEADER_FROM, &from))
	    from = "";
          if (mu_header_sget_value (hdr, MU_HEADER_SUBJECT, &subject))
	    subject = "";
          printf ("%*.*sEncapsulated message : %s\t%s\n",
                  indent, indent, "", from, subject);
          printf ("%*.*sBegin\n", indent, indent, "");
          message_display_parts (part, indent + indent_level);
          mu_message_destroy (&part, NULL);
        }
      else if (!type
               || (mu_c_strcasecmp (type, "text/plain") == 0)
               || (mu_c_strcasecmp (type, "text/html")) == 0)
	{
	  printf ("%*.*sText Message\n", indent, indent, "");
          printf ("%*.*sBegin\n", indent, indent, "");
          mu_message_get_body (part, &body);
          mu_body_get_stream (body, &str);
          /* Make sure the original body stream is not closed when
             str gets destroyed */
          mu_filter_create (&str, str, encoding, MU_FILTER_DECODE,
			    MU_STREAM_READ | MU_STREAM_NO_CLOSE);
          offset = 0;
          while (mu_stream_readline (str, buf, sizeof (buf),
				     offset, &nbytes) == 0 && nbytes)
            {
              printf ("%*.*s%s", indent, indent, "", buf);
              offset += nbytes;
            }
          mu_stream_destroy (&str, NULL);
        }
      else
        {
          /* Save the attachements.  */
          char *fname = NULL;

          mu_message_aget_decoded_attachment_name (part, charset,
						   &fname, NULL);
          if (fname == NULL)
            fname = mu_tempname (NULL);

          printf ("%*.*sAttachment - saving [%s]\n", indent, indent, "",
                  fname);
          printf ("%*.*sBegin\n", indent, indent, "");
          if (charset)
	    {
	      mu_mime_io_buffer_t info;
	      mu_mime_io_buffer_create (&info);
	      mu_mime_io_buffer_set_charset (info, charset);
	      MU_ASSERT (mu_message_save_attachment (part, NULL, info));
	      mu_mime_io_buffer_destroy (&info);
	    }
	  else
	    MU_ASSERT (mu_message_save_attachment (part, fname, NULL));
          if (print_attachments)
            print_file (fname, indent);
          free (fname);
        }
      printf ("\n%*.*sEnd\n", indent, indent, "");
      free (type);
    }
}

