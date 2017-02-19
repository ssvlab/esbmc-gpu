/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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

/* First draft by Alain Magloire */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <errno.h>

#include <mailutils/property.h>
#include <mailutils/stream.h>

#include <filter0.h>

static int rfc822_init (mu_filter_t);
static void rfc822_destroy (mu_filter_t);
static int rfc822_read (mu_filter_t, char *, size_t, mu_off_t, size_t *);
static int rfc822_readline (mu_filter_t, char *, size_t, mu_off_t, size_t *);
static int rfc822_read0 (mu_filter_t, char *, size_t, mu_off_t, size_t *, int);

struct rfc822
{
  mu_off_t r_offset; /* rfc822 offset.  */
  mu_off_t s_offset; /* stream offset.  */
  size_t lines;
  int residue;
};

static struct _mu_filter_record _rfc822_filter =
{
  "RFC822",
  rfc822_init,
  NULL,
  NULL,
  NULL,
};

/* Exported.  */
mu_filter_record_t mu_rfc822_filter = &_rfc822_filter;

static int
rfc822_init (mu_filter_t filter)
{
  mu_property_t property;
  int status;
  filter->data = calloc (1, sizeof (struct rfc822));
  if (filter->data == NULL)
    return ENOMEM;

  filter->_read = rfc822_read;
  filter->_readline = rfc822_readline;
  filter->_destroy = rfc822_destroy;

  /* We are interested in this property.  */
  if ((status = mu_stream_get_property (filter->filter_stream, &property) != 0)
      || (status = mu_property_set_value (property, "LINES", "0", 1)) != 0)
    {
      free (filter->data);
      filter->data = NULL;
      return status;
    }
  return 0;
}

static void
rfc822_destroy (mu_filter_t filter)
{
  if (filter->data)
    free (filter->data);
}

static int
rfc822_read (mu_filter_t filter, char *buffer, size_t buflen,
	      mu_off_t off, size_t *pnread)
{
  return rfc822_read0 (filter, buffer, buflen, off, pnread, 0);
}

static int
rfc822_readline (mu_filter_t filter, char *buffer, size_t buflen,
		 mu_off_t off, size_t *pnread)
{
  return rfc822_read0 (filter, buffer, buflen, off, pnread, 1);
}

/* RFC 822 converter "\n" --> "\r\n"
   We maintain two offsets, the rfc822 offset (r_offset) and the offset of
   the stream (s_offset).  If they do not match we go back as far as possible
   and start to read by 1 'till we reach the current offset.  */

static int
rfc822_read0 (mu_filter_t filter, char *buffer, size_t buflen,
	      mu_off_t off, size_t *pnread, int isreadline)
{
  size_t total = 0;
  int status = 0;
  struct rfc822 *rfc822 = filter->data;

  /* Catch up i.e bring us to the current offset.  */
  if (rfc822->r_offset != off)
    {
      rfc822->residue = 0;
 
      /* Try to find a starting point.  */
      if (rfc822->lines)
	{
	  rfc822->r_offset = off - rfc822->lines;
	  if (rfc822->r_offset < 0)
	    rfc822->r_offset = 0;
	}
      else
	rfc822->r_offset = 0;

      rfc822->s_offset = rfc822->r_offset;

      while (rfc822->r_offset < off)
	{
	  char c;
	  size_t n = 0;
	  status = mu_stream_read (filter->stream, &c, 1, rfc822->s_offset, &n);
	  if (status != 0)
	    return status;
	  if (n == 0)
	    {
	      if (pnread)
		*pnread = 0;
	      return 0;
	    }
	  if (c == '\n')
	    {
	      rfc822->r_offset++;
	      if (rfc822->r_offset == off)
		{
		  rfc822->residue = 1;
		  break;
		}
	    }
	  rfc822->r_offset++;
	  rfc822->s_offset++;
	}
    }

  do
    {
      size_t nread = 0;
      status = mu_stream_readline (filter->stream, buffer, buflen,
				   rfc822->s_offset, &nread);
      if (status != 0)
	return status;
      if (nread == 0)
	break;
      rfc822->r_offset += nread;
      rfc822->s_offset += nread;
      total += nread;
      buflen -= nread;
      if (buffer[nread - 1] == '\n')
	{
	  if (!rfc822->residue)
	    {
	      buffer[nread - 1] = '\r';
	      if (buflen == 0)
		{
		  rfc822->residue = 1;
		  break;
		}
	      buffer[nread] = '\n';
	      buflen--;
	      nread++;
	      total++;
	      rfc822->r_offset++;
	    }
	  else
	    rfc822->residue = 0;
	}
      buffer += nread;
    } while (buflen > 0 && !isreadline);

  if (isreadline && buffer)
    *buffer = '\0';

  if (pnread)
    *pnread = total;
  return status;
}
