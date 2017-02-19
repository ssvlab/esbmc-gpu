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

#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <mailutils/sys/pop3.h>

/* Implementation of the stream for TOP and RETR.  */
struct mu_pop3_stream
{
  mu_pop3_t pop3;
  int done;
};

static void
mu_pop3_stream_destroy (mu_stream_t stream)
{
  struct mu_pop3_stream *pop3_stream = mu_stream_get_owner (stream);
  if (pop3_stream)
    {
      free (pop3_stream);
    }
}

static int
mu_pop3_stream_read (mu_stream_t stream, char *buf, size_t buflen, mu_off_t offset, size_t *pn)
{
  struct mu_pop3_stream *pop3_stream = mu_stream_get_owner (stream);
  size_t n = 0;
  int status = 0;
  char *p = buf;

  (void)offset;
  if (pop3_stream)
    {
      if (!pop3_stream->done)
	{
	  do
	    {
	      size_t nread = 0;

	      /* The pop3_readline () function will always read one less to
		 be able to null terminate the buffer, this will cause
		 serious grief for mu_stream_read() where it is legitimate to
		 have a buffer of 1 char.  So we must catch it here.  */
	      if (buflen == 1)
		{
		  char buffer[2];
		  *buffer = '\0';
		  status = mu_pop3_readline (pop3_stream->pop3, buffer, 2, &nread);
		  *p = *buffer;
		}
	      else
		status = mu_pop3_readline (pop3_stream->pop3, p, buflen, &nread);

	      if (status != 0)
		break;
	      if (nread == 0)
		{
		  pop3_stream->pop3->state = MU_POP3_NO_STATE;
		  pop3_stream->done = 1;
		  break;
		}
	      n += nread;
	      buflen -= nread;
	      p += nread;
	    }
	  while (buflen > 0);
	}
    }
  if (pn)
    *pn = n;
  return status;
}

static int
mu_pop3_stream_readline (mu_stream_t stream, char *buf, size_t buflen, mu_off_t offset, size_t *pn)
{
  struct mu_pop3_stream *pop3_stream = mu_stream_get_owner (stream);
  size_t n = 0;
  int status = 0;

  (void)offset;
  if (pop3_stream)
    {
      if (!pop3_stream->done)
	{
	  status = mu_pop3_readline (pop3_stream->pop3, buf, buflen, &n);
	  if (n == 0)
	    {
	      pop3_stream->pop3->state = MU_POP3_NO_STATE;
	      pop3_stream->done = 1;
	    }
	}
    }
  if (pn)
    *pn = n;
  return status;
}

int
mu_pop3_stream_create (mu_pop3_t pop3, mu_stream_t *pstream)
{
  struct mu_pop3_stream *pop3_stream;
  int status;

  pop3_stream = malloc (sizeof *pop3_stream);
  if (pop3_stream == NULL)
    return ENOMEM;

  pop3_stream->pop3 = pop3;
  pop3_stream->done = 0;

  status = mu_stream_create (pstream, MU_STREAM_READ | MU_STREAM_NO_CLOSE | MU_STREAM_NO_CHECK, pop3_stream);
  if (status != 0)
    {
      free (pop3_stream);
      return status;
    }
 
  mu_stream_set_read (*pstream, mu_pop3_stream_read, pop3_stream);
  mu_stream_set_readline (*pstream, mu_pop3_stream_readline, pop3_stream);
  mu_stream_set_destroy (*pstream, mu_pop3_stream_destroy, pop3_stream);
                                                                                                                             
  return 0;
}
