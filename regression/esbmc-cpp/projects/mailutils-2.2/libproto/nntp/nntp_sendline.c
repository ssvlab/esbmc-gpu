/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include <errno.h>
#include <mailutils/sys/nntp.h>

/* A socket may write less then expected but stream.c:mu_stream_write() will
   always try to send the entire buffer unless an error is reported.  We have
   to cope with nonblocking, it is done by keeping track with the nntp->ptr
   pointer if the write failed we keep track and restart where we left.  */
int
mu_nntp_send (mu_nntp_t nntp)
{
  int status = 0;
  if (nntp->carrier && (nntp->io.ptr > nntp->io.buf))
    {
      size_t n = 0;
      size_t len = nntp->io.ptr - nntp->io.buf;

      /* Timeout with select(), note that we have to reset select()
	 since on linux tv is modified when error.  */
      if (nntp->timeout)
	{
	  int ready = mu_nntp_carrier_is_ready (nntp->carrier,
						MU_STREAM_READY_WR,
						nntp->timeout);
	  if (ready == 0)
	    return ETIMEDOUT;
	}

      status = mu_stream_write (nntp->carrier, nntp->io.buf, len, 0, &n);
      if (n)
	{
	  /* Consume what we sent.  */
	  memmove (nntp->io.buf, nntp->io.buf + n, len - n);
	  nntp->io.ptr -= n;
	}
    }
  else
    nntp->io.ptr = nntp->io.buf;
  return status;
}

/* According to RFC 2449: The maximum length of a command is increased from
   47 characters (4 character command, single space, 40 character argument,
   CRLF) to 255 octets, including the terminating CRLF.  But we are flexible
   on this and realloc() as needed. NOTE: The terminated CRLF is not
   included.  */
int
mu_nntp_writeline (mu_nntp_t nntp, const char *format, ...)
{
  int len;
  va_list ap;
  int done = 1;

  va_start(ap, format);
  /* C99 says that a conforming implementation of snprintf () should
     return the number of char that would have been call but many old
     GNU/Linux && BSD implementations return -1 on error.  Worse,
     QnX/Neutrino actually does not put the terminal null char.  So
     let's try to cope.  */
  do
    {
      len = vsnprintf (nntp->io.buf, nntp->io.len - 1, format, ap);
      if (len < 0 || len >= (int)nntp->io.len
	  || !memchr (nntp->io.buf, '\0', len + 1))
	{
	  nntp->io.len *= 2;
	  nntp->io.buf = realloc (nntp->io.buf, nntp->io.len);
	  if (nntp->io.buf == NULL)
	    return ENOMEM;
	  done = 0;
	}
      else
	done = 1;
    }
  while (!done);
  va_end(ap);
  nntp->io.ptr = nntp->io.buf + len;
  return 0;
}

int
mu_nntp_sendline (mu_nntp_t nntp, const char *line)
{
  if (line)
    {
      int status = mu_nntp_writeline (nntp, line);
      if (status)
	return status;
    }
  return mu_nntp_send (nntp);
}
