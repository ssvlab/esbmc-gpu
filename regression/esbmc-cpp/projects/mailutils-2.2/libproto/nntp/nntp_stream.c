/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2007, 2010 Free Software Foundation, Inc.

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
#include <mailutils/sys/nntp.h>

/* Implementation of the stream for HELP, ARTICLE, etc ...  */
struct mu_nntp_stream
{
  mu_nntp_t nntp;
  int done;
};

static void
mu_nntp_stream_destroy (mu_stream_t stream)
{
  struct mu_nntp_stream *nntp_stream = mu_stream_get_owner (stream);
  if (nntp_stream)
    {
      free (nntp_stream);
    }
}

static int
mu_nntp_stream_read (mu_stream_t stream, char *buf, size_t buflen, mu_off_t offset, size_t *pn)
{
  struct mu_nntp_stream *nntp_stream = mu_stream_get_owner (stream);
  size_t n = 0;
  int status = 0;
  char *p = buf;

  (void)offset;
  if (nntp_stream)
    {
      if (!nntp_stream->done)
	{
	  do
	    {
	      size_t nread = 0;

	      /* The nntp_readline () function will always read one less to
		 be able to null terminate the buffer, this will cause
		 serious grief for mu_stream_read() where it is legitimate to
		 have a buffer of 1 char.  So we must catch it here.  */
	      if (buflen == 1)
		{
		  char buffer[2];
		  *buffer = '\0';
		  status = mu_nntp_readline (nntp_stream->nntp, buffer, 2, &nread);
		  *p = *buffer;
		}
	      else
		status = mu_nntp_readline (nntp_stream->nntp, p, buflen, &nread);

	      if (status != 0)
		break;
	      if (nread == 0)
		{
		  nntp_stream->nntp->state = MU_NNTP_NO_STATE;
		  nntp_stream->done = 1;
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
mu_nntp_stream_readline (mu_stream_t stream, char *buf, size_t buflen, mu_off_t offset, size_t *pn)
{
  struct mu_nntp_stream *nntp_stream = mu_stream_get_owner (stream);
  size_t n = 0;
  int status = 0;

  (void)offset;
  if (nntp_stream)
    {
      if (!nntp_stream->done)
	{
	  status = mu_nntp_readline (nntp_stream->nntp, buf, buflen, &n);
	  if (n == 0)
	    {
	      nntp_stream->nntp->state = MU_NNTP_NO_STATE;
	      nntp_stream->done = 1;
	    }
	}
    }
  if (pn)
    *pn = n;
  return status;
}

int
mu_nntp_stream_create (mu_nntp_t nntp, mu_stream_t *pstream)
{
  struct mu_nntp_stream *nntp_stream;
  int status;

  nntp_stream = malloc (sizeof *nntp_stream);
  if (nntp_stream == NULL)
    return ENOMEM;

  nntp_stream->nntp = nntp;
  nntp_stream->done = 0;

  status = mu_stream_create (pstream, MU_STREAM_READ | MU_STREAM_NO_CLOSE | MU_STREAM_NO_CHECK, nntp_stream);
  if (status != 0)
    {
      free (nntp_stream);
      return status;
    }

  mu_stream_set_read (*pstream, mu_nntp_stream_read, nntp_stream);
  mu_stream_set_readline (*pstream, mu_nntp_stream_readline, nntp_stream);
  mu_stream_set_destroy (*pstream, mu_nntp_stream_destroy, nntp_stream);

  return 0;
}
