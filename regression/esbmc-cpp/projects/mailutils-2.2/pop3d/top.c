/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "pop3d.h"

/* Prints the header of a message plus a specified number of lines.  */

int
pop3d_top (char *arg)
{
  size_t mesgno;
  int lines;
  mu_message_t msg;
  mu_attribute_t attr;
  mu_header_t hdr;
  mu_body_t body;
  mu_stream_t stream;
  char *mesgc, *linesc;
  char buf[BUFFERSIZE];
  size_t n;
  off_t off;

  if (strlen (arg) == 0)
    return ERR_BAD_ARGS;

  if (state != TRANSACTION)
    return ERR_WRONG_STATE;

  pop3d_parse_command (arg, &mesgc, &linesc);
  
  mesgno = strtoul (mesgc, NULL, 10);
  lines = *linesc ? strtol (linesc, NULL, 10) : -1;

  if (lines < 0)
    return ERR_BAD_ARGS;

  if (mu_mailbox_get_message (mbox, mesgno, &msg) != 0)
    return ERR_NO_MESG;

  mu_message_get_attribute (msg, &attr);
  if (pop3d_is_deleted (attr))
    return ERR_MESG_DELE;
  pop3d_mark_retr (attr);
  
  pop3d_outf ("+OK\r\n");

  /* Header.  */
  mu_message_get_header (msg, &hdr);
  mu_header_get_stream (hdr, &stream);
  off = n = 0;
  while (mu_stream_readline (stream, buf, sizeof(buf), off, &n) == 0
	 && n > 0)
    {
      /* Nuke the trainline newline.  */
      if (buf[n - 1] == '\n')
	{
	  buf[n - 1] = '\0';
	  pop3d_outf ("%s\r\n", buf);
	}
      else
	pop3d_outf ("%s", buf);
      off += n;
    }

  /* Lines of body.  */
  if (lines)
    {
      int prev_nl = 1;

      mu_message_get_body (msg, &body);
      mu_body_get_stream (body, &stream);
      n = off = 0;
      while (mu_stream_readline (stream, buf, sizeof(buf), off, &n) == 0
	     && n > 0 && lines > 0)
	{
	  if (prev_nl && buf[0] == '.')
	    pop3d_outf (".");
      
	  if (buf[n - 1] == '\n')
	    {
	      buf[n - 1] = '\0';
	      pop3d_outf ("%s\r\n", buf);
	      prev_nl = 1;
	      lines--;
	    }
	  else
	    {
	      pop3d_outf ("%s", buf);
	      prev_nl = 0;
	    }
	  off += n;
	}
      if (!prev_nl)
	pop3d_outf ("\r\n");
    }

  pop3d_outf (".\r\n");

  return OK;
}
