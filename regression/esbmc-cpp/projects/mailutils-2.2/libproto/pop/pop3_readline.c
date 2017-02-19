/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2010 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <mailutils/sys/pop3.h>
#include <mailutils/error.h>

int
mu_pop3_carrier_is_ready (mu_stream_t carrier, int flag, int timeout)
{
  struct timeval tv, *tvp = NULL;
  int wflags = flag;
  int status;
  
  if (timeout >= 0)
    {
      tv.tv_sec  = timeout / 100;
      tv.tv_usec = (timeout % 1000) * 1000;
      tvp = &tv;
    }
  status = mu_stream_wait (carrier, &wflags, tvp);
  if (status)
    return 0; /* FIXME: provide a way to return error code! */
  return wflags & flag;
}

/* Read a complete line from the pop server. Transform CRLF to LF, remove
   the stuff byte termination octet ".", put a null in the buffer
   when done.  And Do a select() (stream_is_readready()) for the timeout.  */
static int
mu_pop3_getline (mu_pop3_t pop3)
{
  size_t n = 0;
  size_t total = pop3->io.ptr - pop3->io.buf;
  int status = 0;

  /* Must get a full line before bailing out.  */
  do
    {
      /* Timeout with select(), note that we have to reset select()
	 since on linux tv is modified when error.  */
      if (pop3->timeout)
	{
	  int ready = mu_pop3_carrier_is_ready (pop3->carrier,
						MU_STREAM_READY_RD,
						pop3->timeout);
	  if (ready == 0)
	    return ETIMEDOUT;
	}

      status = mu_stream_sequential_readline (pop3->carrier, pop3->io.buf + total, pop3->io.len - total, &n);
      if (status != 0)
	return status;

      /* The server went away:  It maybe a timeout and some pop server
	 does not send the -ERR.  Consider this like an error.  */
      if (n == 0)
	return EIO;

      total += n;
      pop3->io.nl = memchr (pop3->io.buf, '\n', total);
      if (pop3->io.nl == NULL)  /* Do we have a full line.  */
	{
	  /* Allocate a bigger buffer ?  */
	  if (total >= pop3->io.len - 1)
	    {
	      pop3->io.len *= 2;
	      pop3->io.buf = realloc (pop3->io.buf, pop3->io.len + 1);
	      if (pop3->io.buf == NULL)
		return ENOMEM;
	    }
	}
      pop3->io.ptr = pop3->io.buf + total;
    }
  while (pop3->io.nl == NULL); /* Bail only if we have a complete line.  */

  /* When examining a multi-line response, the client checks to see if the
     line begins with the termination octet "."(DOT). If yes and if octets
     other than CRLF follow, the first octet of the line (the termination
     octet) is stripped away.  */
  if (total >= 3  && pop3->io.buf[0] == '.')
    {
      if (pop3->io.buf[1] != '\r' && pop3->io.buf[2] != '\n')
	{
	  memmove (pop3->io.buf, pop3->io.buf + 1, total - 1);
	  pop3->io.ptr--;
	  pop3->io.nl--;
	}
      /* And if CRLF immediately follows the termination character, then
	 the response from the POP server is ended and the line containing
	 ".CRLF" is not considered part of the multi-line response.  */
      else if (pop3->io.buf[1] == '\r' && pop3->io.buf[2] == '\n')
	{
	  pop3->io.buf[0] = '\0';
	  pop3->io.ptr = pop3->io.buf;
	  pop3->io.nl = NULL;
	}
    }
  /* \r\n --> \n\0, conversion.  */
  if (pop3->io.nl > pop3->io.buf)
    {
      *(pop3->io.nl - 1) = '\n';
      *(pop3->io.nl) = '\0';
      pop3->io.ptr = pop3->io.nl;
    }
  return status;
}

/* Call pop3_getline() for the dirty work,  and consume i.e. put
   in the user buffer only buflen. If buflen == 0 or buffer == NULL
   nothing is consume, the data is save for another call to pop3_readline()
   with a buffer != NULL.
  */
int
mu_pop3_readline (mu_pop3_t pop3, char *buffer, size_t buflen, size_t *pnread)
{
  size_t nread = 0;
  size_t n = 0;
  int status = 0;

  /* Do we need to fill up? Yes if no NL or the buffer is empty.  */
  if (pop3->carrier && (pop3->io.nl == NULL || pop3->io.ptr == pop3->io.buf))
    {
      status = mu_pop3_getline (pop3);
      if (status != 0)
	return status;
    }

  /* How much we can copy ?  */
  n = pop3->io.ptr - pop3->io.buf;

  /* Consume the line?  */
  if (buffer && buflen)
    {
      buflen--; /* For the null.  */
      if (buflen)
	{
	  int nleft = buflen - n;
	  /* We got more then requested.  */
	  if (nleft < 0)
	    {
	      size_t sentinel;
	      nread = buflen;
	      sentinel = pop3->io.ptr - (pop3->io.buf + nread);
	      memcpy (buffer, pop3->io.buf, nread);
	      memmove (pop3->io.buf, pop3->io.buf + nread, sentinel);
	      pop3->io.ptr = pop3->io.buf + sentinel;
	    }
	  else
	    {
	      /* Drain our buffer.  */;
	      nread = n;
	      memcpy (buffer, pop3->io.buf, nread);
	      pop3->io.ptr = pop3->io.buf;
	      /* Clear of all residue.  */
	      memset (pop3->io.buf, '\0', pop3->io.len);
	    }
	}
      buffer[nread] = '\0';
    }
  else
    nread = n;

  if (pnread)
    *pnread = nread;
  return status;
}
