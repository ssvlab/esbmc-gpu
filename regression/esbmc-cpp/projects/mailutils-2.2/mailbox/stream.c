/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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


/* Credits.  Some of the Readline an buffering scheme was taken
   from 4.4BSDLite2.

   Copyright (c) 1990, 1993
   The Regents of the University of California.  All rights reserved.

   This code is derived from software contributed to Berkeley by
   Chris Torek.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <mailutils/property.h>
#include <mailutils/errno.h>
#include <mailutils/io.h>
#include <stream0.h>

static int refill (mu_stream_t, mu_off_t);

/* A stream provides a way for an object to do I/O. It overloads
   stream read/write functions. Only a minimal buffering is done
   and that if stream's bufsiz member is set. If the requested
   offset does not equal the one maintained internally the buffer
   is flushed and refilled. This buffering scheme is more convenient
   for networking streams (POP/IMAP).
   Writes are always unbuffered. */
int
mu_stream_create (mu_stream_t *pstream, int flags, void *owner)
{
  mu_stream_t stream;
  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (owner == NULL)
    return EINVAL;
  stream = calloc (1, sizeof (*stream));
  if (stream == NULL)
    return ENOMEM;
  stream->owner = owner;
  stream->flags = flags;
  /* By default unbuffered, the buffering scheme is not for all models, it
     really makes sense for network streams, where there is no offset.  */
  /* stream->rbuffer.bufsiz = BUFSIZ; */
  *pstream = stream;
  return 0;
}

void
mu_stream_destroy (mu_stream_t *pstream, void *owner)
{
   if (pstream && *pstream)
    {
      mu_stream_t stream = *pstream;
      if ((stream->flags & MU_STREAM_NO_CHECK) || stream->owner == owner)
	{
	  mu_stream_close (stream);
	  if (stream->rbuffer.base)
	    free (stream->rbuffer.base);

	  if (stream->_destroy)
	    stream->_destroy (stream);

	  free (stream);
	}
      *pstream = NULL;
    }
}

void *
mu_stream_get_owner (mu_stream_t stream)
{
  return (stream) ? stream->owner : NULL;
}

int
mu_stream_open (mu_stream_t stream)
{
  if (stream == NULL)
    return EINVAL;
  stream->state = MU_STREAM_STATE_OPEN;

  if (stream->_open)
    return stream->_open (stream);
  return  0;
}

int
mu_stream_close (mu_stream_t stream)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->state == MU_STREAM_STATE_CLOSE)
    return 0;

  stream->state = MU_STREAM_STATE_CLOSE;
  /* Clear the buffer of any residue left.  */
  if (stream->rbuffer.base)
    {
      stream->rbuffer.ptr = stream->rbuffer.base;
      stream->rbuffer.count = 0;
      memset (stream->rbuffer.base, '\0', stream->rbuffer.bufsiz);
    }
  if (stream->_close)
    return stream->_close (stream);
  return  0;
}

int
mu_stream_is_seekable (mu_stream_t stream)
{
  return (stream) ? stream->flags & MU_STREAM_SEEKABLE : 0;
}

int
mu_stream_setbufsiz (mu_stream_t stream, size_t size)
{
  if (stream == NULL)
    return EINVAL;
  stream->rbuffer.bufsiz = size;
  return 0;
}

/* We have to be clear about the buffering scheme, it is not designed to be
   used as a full-fledged buffer mechanism.  It is a simple mechanism for
   networking. Lots of code between POP and IMAP can be shared this way.
   - First caveat; the code maintains its own offset (rbuffer.offset member)
   and if it does not match the requested one, the data is flushed
   and the underlying _read is called. It is up to the latter to return
   EISPIPE when appropriate.
   - Again, this is targeting networking stream to make readline()
   a little bit more efficient, instead of reading a char at a time.  */

int
mu_stream_read (mu_stream_t is, char *buf, size_t count,
		mu_off_t offset, size_t *pnread)
{
  int status = 0;
  if (is == NULL || is->_read == NULL)
    return EINVAL;

  is->state = MU_STREAM_STATE_READ;

  /* Sanity check; noop.  */
  if (count == 0)
    {
      if (pnread)
	*pnread = 0;
      return 0;
    }

  /* If rbuffer.bufsiz == 0.  It means they did not want the buffer
     mechanism.  Good for them.  */
  if (is->rbuffer.bufsiz == 0)
    status = is->_read (is, buf, count, offset, pnread);
  else
    {
      size_t residue = count;
      size_t r;

      /* If the amount requested is bigger than the buffer cache size,
	 bypass it.  Do no waste time and let it through.  */
      if (count > is->rbuffer.bufsiz)
	{
	  r = 0;
	  /* Drain the buffer first.  */
	  if (is->rbuffer.count > 0 && offset == is->rbuffer.offset)
	    {
	      memcpy(buf, is->rbuffer.ptr, is->rbuffer.count);
	      is->rbuffer.offset += is->rbuffer.count;
	      residue -= is->rbuffer.count;
	      buf += is->rbuffer.count;
	      offset += is->rbuffer.count;
	    }
	  is->rbuffer.count = 0;
	  status = is->_read (is, buf, residue, offset, &r);
	  is->rbuffer.offset += r;
	  residue -= r;
	  if (pnread)
	    *pnread = count - residue;
	  return status;
	}

      /* Fill the buffer, do not want to start empty hand.  */
      if (is->rbuffer.count <= 0 || offset != is->rbuffer.offset)
	{
	  status = refill (is, offset);
	  if (status != 0)
	    return status;
	  /* Reached the end ??  */
	  if (is->rbuffer.count == 0)
	    {
	      if (pnread)
		*pnread = 0;
	      return status;
	    }
	}

      /* Drain the buffer, if we have less then requested.  */
      while (residue > (size_t)(r = is->rbuffer.count))
	{
	  memcpy (buf, is->rbuffer.ptr, (size_t)r);
	  /* stream->rbuffer.count = 0 ... done in refill */
	  is->rbuffer.ptr += r;
	  is->rbuffer.offset += r;
	  buf += r;
	  residue -= r;
	  status = refill (is, is->rbuffer.offset);
	  if (status != 0)
	    {
	      /* We have something in the buffer return the error on the
		 next call .  */
	      if (count != residue)
		{
		  if (pnread)
		    *pnread = count - residue;
		  status = 0;
		}
	      return status;
	    }
	  /* Did we reach the end.  */
	  if (is->rbuffer.count == 0)
	    {
	      if (pnread)
		*pnread = count - residue;
	      return status;
	    }
	}
      memcpy(buf, is->rbuffer.ptr, residue);
      is->rbuffer.count -= residue;
      is->rbuffer.ptr += residue;
      is->rbuffer.offset += residue;
      if (pnread)
	*pnread = count;
    }
  return status;
}

/*
 * Read at most n-1 characters.
 * Stop when a newline has been read, or the count runs out.
 */
int
mu_stream_readline (mu_stream_t is, char *buf, size_t count,
		    mu_off_t offset, size_t *pnread)
{
  int status = 0;

  if (is == NULL)
    return EINVAL;

  is->state = MU_STREAM_STATE_READ;

  switch (count)
    {
    case 1:
      /* why would they do a thing like that?
	 mu_stream_readline() is __always null terminated.  */
      if (buf)
	*buf = '\0';
    case 0: /* Buffer is empty noop.  */
      if (pnread)
	*pnread = 0;
      return 0;
    }

  /* Use the provided readline.  */
  if (is->rbuffer.bufsiz == 0 &&  is->_readline != NULL)
    status = is->_readline (is, buf, count, offset, pnread);
  else if (is->rbuffer.bufsiz == 0) /* No Buffering.  */
    {
      size_t n, nr = 0;
      char c;
      /* Grossly inefficient hopefully they override this */
      count--;  /* Leave space for the null.  */
      for (n = 0; n < count; )
	{
	  status = is->_read (is, &c, 1, offset, &nr);
	  if (status != 0) /* Error.  */
	    return status;
	  else if (nr == 1)
	    {
	      *buf++ = c;
	      offset++;
	      n++;
	      if (c == '\n') /* Newline is stored like fgets().  */
		break;
	    }
	  else if (nr == 0)
	    break; /* EOF */
	}
      *buf = '\0';
      if (pnread)
	*pnread = n;
    }
  else /* Buffered.  */
    {
      char *s = buf;
      char *p, *nl;
      size_t len;
      size_t total = 0;

      count--;  /* Leave space for the null.  */

      /* If out of range refill.  */
      /*      if ((offset < is->rbuffer.offset */
      /*	   || offset > (is->rbuffer.offset + is->rbuffer.count))) */
      if (offset != is->rbuffer.offset)
	{
	  status = refill (is, offset);
	  if (status != 0)
	    return status;
	  if (is->rbuffer.count == 0)
	    {
	      if (pnread)
		*pnread = 0;
	      return 0;
	    }
	}

      while (count != 0)
	{
	  /* If the buffer is empty refill it.  */
	  len = is->rbuffer.count;
	  if (len <= 0)
	    {
	      status = refill (is, is->rbuffer.offset);
	      if (status != 0)
		{
		  if (s != buf)
		    break;
		}
	      len = is->rbuffer.count;
	      if (len == 0)
		break;
	    }
	  p = is->rbuffer.ptr;

	  /* Scan through at most n bytes of the current buffer,
	     looking for '\n'.  If found, copy up to and including
	     newline, and stop.  Otherwise, copy entire chunk
	     and loop.  */
	  if (len > count)
	    len = count;
	  nl = memchr ((void *)p, '\n', len);
	  if (nl != NULL)
	    {
	      len = ++nl - p;
	      is->rbuffer.count -= len;
	      is->rbuffer.ptr = nl;
	      is->rbuffer.offset += len;
	      memcpy ((void *)s, (void *)p, len);
	      total += len;
	      s[len] = 0;
	      if (pnread)
		*pnread = total;
	      return 0;
	    }
	  is->rbuffer.count -= len;
	  is->rbuffer.ptr += len;
	  is->rbuffer.offset += len;
	  memcpy((void *)s, (void *)p, len);
	  total += len;
	  s += len;
	  count -= len;
        }
      *s = 0;
      if (pnread)
	*pnread = s - buf;
    }
  return status;
}

int
mu_stream_getline (mu_stream_t is, char **pbuf, size_t *pbufsize,
		   mu_off_t offset, size_t *pnread)
{
  char *buf = *pbuf;
  size_t bufsize = *pbufsize;
  size_t total = 0, off = 0;
  int rc = 0;
#define DELTA 128
  
  if (buf == NULL)
    {
      bufsize = DELTA;
      buf = malloc (bufsize);
      if (!buf)
	return ENOMEM;
    }

  do
    {
      size_t nread;
      int rc;

      if (off == bufsize)
	{
	  char *p;
	  p = realloc (buf, bufsize + DELTA);
	  if (!p)
	    {
	      rc = ENOMEM;
	      break;
	    }
	  bufsize += DELTA;
	  buf = p;
	}
      
      rc = mu_stream_readline (is, buf + off, bufsize - off, offset + off,
			       &nread);
      if (rc)
	{
	  if (*pbuf)
	    free (buf);
	  return rc;
	}
      if (nread == 0)
	break;
      off += nread;
      total += nread;
    }
  while (buf[off - 1] != '\n');

  if (rc && !*pbuf)
    free (buf);
  else
    {
      *pbuf = buf;
      *pbufsize = bufsize;
      if (pnread)
	*pnread = total;
    }
  return rc;
}

int
mu_stream_write (mu_stream_t os, const char *buf, size_t count,
		 mu_off_t offset, size_t *pnwrite)
{
  int nleft;
  int err = 0;
  size_t nwriten = 0;
  size_t total = 0;

  if (os == NULL || os->_write == NULL)
      return EINVAL;
  os->state = MU_STREAM_STATE_WRITE;

  nleft = count;
  /* First try to send it all.  */
  while (nleft > 0)
    {
      err = os->_write (os, buf, nleft, offset, &nwriten);
      if (err != 0 || nwriten == 0)
        break;
      nleft -= nwriten;
      total += nwriten;
      buf += nwriten;
    }
  if (pnwrite)
    *pnwrite = total;
  return err;
}

int
mu_stream_vprintf (mu_stream_t os, mu_off_t *poff, const char *fmt, va_list ap)
{
  char *buf = NULL, *p;
  size_t buflen = 0;
  size_t n;
  int rc;

  rc = mu_vasnprintf (&buf, &buflen, fmt, ap);
  if (rc)
    return rc;
  p = buf;
  n = strlen (buf);
  do
    {
      size_t wrs;

      rc = mu_stream_write (os, p, n, *poff, &wrs);
      if (rc || wrs == 0)
        break;
      p += wrs;
      *poff += wrs;
      n -= wrs;
    }
  while (n > 0);
  free (buf);
  return rc;
}

int
mu_stream_printf (mu_stream_t os, mu_off_t *poff, const char *fmt, ...)
{ 
  va_list ap;
  int rc;
	
  va_start (ap, fmt);
  rc = mu_stream_vprintf (os, poff, fmt, ap);
  va_end (ap);
  return rc;
}

int
mu_stream_sequential_vprintf (mu_stream_t os, const char *fmt, va_list ap)
{
  return mu_stream_vprintf (os, &os->offset, fmt, ap);
}

int
mu_stream_sequential_printf (mu_stream_t os, const char *fmt, ...)
{
  va_list ap;
  int rc;
	
  va_start (ap, fmt);
  rc = mu_stream_sequential_vprintf (os, fmt, ap);
  va_end (ap);
  return rc;
}

int
mu_stream_get_transport2 (mu_stream_t stream,
			  mu_transport_t *p1, mu_transport_t *p2)
{
  if (stream == NULL || stream->_get_transport2 == NULL)
    return EINVAL;
  return stream->_get_transport2 (stream, p1, p2);
}

int
mu_stream_get_transport (mu_stream_t stream, mu_transport_t *pt)
{
  return mu_stream_get_transport2 (stream, pt, NULL);
}

int
mu_stream_get_flags (mu_stream_t stream, int *pfl)
{
  if (stream == NULL)
    return EINVAL;
  if (pfl == NULL)
    return MU_ERR_OUT_NULL;
  *pfl = stream->flags;
  return 0;
}

int
mu_stream_set_property (mu_stream_t stream, mu_property_t property, void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  if (stream->property)
    mu_property_destroy (&(stream->property), stream);
  stream->property = property;
  return 0;
}

int
mu_stream_get_property (mu_stream_t stream, mu_property_t *pp)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->property == NULL)
    {
      int status = mu_property_create (&(stream->property), stream);
      if (status != 0)
	return status;
    }
  *pp = stream->property;
  return 0;
}

int
mu_stream_size (mu_stream_t stream, mu_off_t *psize)
{
  if (stream == NULL || stream->_size == NULL)
    return EINVAL;
  return stream->_size (stream, psize);
}

int
mu_stream_truncate (mu_stream_t stream, mu_off_t len)
{
  if (stream == NULL || stream->_truncate == NULL )
    return EINVAL;

  return stream->_truncate (stream, len);
}


int
mu_stream_flush (mu_stream_t stream)
{
  if (stream == NULL || stream->_flush == NULL)
    return EINVAL;
  return stream->_flush (stream);
}


int
mu_stream_get_state (mu_stream_t stream, int *pstate)
{
  if (stream == NULL)
    return EINVAL;
  if (pstate == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *pstate = stream->state;
  return 0;
}

int
mu_stream_shutdown (mu_stream_t stream, int how)
{
  if (stream == NULL)
    return EINVAL;
  if (!stream->_shutdown)
    return ENOSYS;
  switch (how)
    {
    case MU_STREAM_READ:
    case MU_STREAM_WRITE:
      break;

    default:
      return EINVAL;
    }
  return stream->_shutdown (stream, how);
}

int
mu_stream_set_destroy (mu_stream_t stream,
		       void (*_destroy) (mu_stream_t), void *owner)
{
  if (stream == NULL)
    return EINVAL;

  if (stream->owner != owner)
    return EACCES;

  stream->_destroy = _destroy;
  return 0;
}

int
mu_stream_set_open (mu_stream_t stream,
		    int (*_open) (mu_stream_t), void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_open = _open;
      return 0;
    }
  return EACCES;
}

int
mu_stream_set_close (mu_stream_t stream,
		     int (*_close) (mu_stream_t), void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_close = _close;
      return 0;
    }
  return EACCES;
}

int
mu_stream_set_get_transport2 (mu_stream_t stream,
			      int (*_get_trans) (mu_stream_t,
						 mu_transport_t *,
						 mu_transport_t *),
			      void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_get_transport2 = _get_trans;
      return 0;
    }
  return EACCES;
}

int
mu_stream_set_read (mu_stream_t stream,
		    int (*_read) (mu_stream_t, char *, size_t,
				  mu_off_t, size_t *),
		    void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_read = _read;
      return 0;
    }
  return EACCES;
}

int
mu_stream_set_readline (mu_stream_t stream,
			int (*_readline) (mu_stream_t, char *, size_t,
					  mu_off_t, size_t *),
			void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_readline = _readline;
      return 0;
    }
  return EACCES;
}

int
mu_stream_set_write (mu_stream_t stream,
		     int (*_write) (mu_stream_t, const char *, size_t,
				    mu_off_t, size_t *),
		     void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner == owner)
    {
      stream->_write = _write;
      return 0;
    }
  return EACCES;
}


int
mu_stream_set_size (mu_stream_t stream,
		    int (*_size) (mu_stream_t, mu_off_t *),
		    void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  stream->_size = _size;
  return 0;
}

int
mu_stream_set_truncate (mu_stream_t stream,
			int (*_truncate) (mu_stream_t, mu_off_t),
			void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  stream->_truncate = _truncate;
  return 0;
}

int
mu_stream_set_flush (mu_stream_t stream,
		     int (*_flush) (mu_stream_t), void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  stream->_flush = _flush;
  return 0;
}

int
mu_stream_set_flags (mu_stream_t stream, int fl)
{
  if (stream == NULL)
    return EINVAL;
  stream->flags |= fl;
  return 0;
}

int
mu_stream_clr_flags (mu_stream_t stream, int fl)
{
  if (stream == NULL)
    return EINVAL;
  stream->flags &= ~fl;
  return 0;
}

int
mu_stream_set_strerror (mu_stream_t stream,
			int (*fp) (mu_stream_t, const char **), void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  stream->_strerror = fp;
  return 0;
}

int
mu_stream_set_wait (mu_stream_t stream,
		    int (*wait) (mu_stream_t, int *, struct timeval *),
		    void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->owner != owner)
    return EACCES;
  stream->_wait = wait;
  return 0;
}

int
mu_stream_set_shutdown (mu_stream_t stream,
			int (*_shutdown) (mu_stream_t, int how), void *owner)
{
  if (stream == NULL)
    return EINVAL;
  if (owner == stream->owner)
    {
      stream->_shutdown = _shutdown;
      return 0;
    }
  return EACCES;
}

int
mu_stream_sequential_read (mu_stream_t stream, char *buf, size_t size,
			   size_t *nbytes)
{
  size_t rdbytes;
  int rc = mu_stream_read (stream, buf, size, stream->offset, &rdbytes);
  if (!rc)
    {
      stream->offset += rdbytes;
      if (nbytes)
	*nbytes = rdbytes;
    }
  return rc;
}

int
mu_stream_sequential_readline (mu_stream_t stream, char *buf, size_t size,
			       size_t *nbytes)
{
  size_t rdbytes;
  int rc = mu_stream_readline (stream, buf, size, stream->offset, &rdbytes);
  if (!rc)
    {
      stream->offset += rdbytes;
      if (nbytes)
	*nbytes = rdbytes;
    }
  return rc;
}

int
mu_stream_sequential_getline  (mu_stream_t stream,
			       char **pbuf, size_t *pbufsize,
			       size_t *nbytes)
{
  size_t rdbytes;
  int rc = mu_stream_getline (stream, pbuf, pbufsize, stream->offset, &rdbytes);
  if (!rc)
    {
      stream->offset += rdbytes;
      if (nbytes)
	*nbytes = rdbytes;
    }
  return rc;
}  


int
mu_stream_sequential_write (mu_stream_t stream, const char *buf, size_t size)
{
  if (stream == NULL)
    return EINVAL;
  while (size > 0)
    {
      size_t sz;
      int rc = mu_stream_write (stream, buf, size, stream->offset, &sz);
      if (rc)
	return rc;

      buf += sz;
      size -= sz;
      stream->offset += sz;
    }
  return 0;
}

int
mu_stream_seek (mu_stream_t stream, mu_off_t off, int whence)
{
  mu_off_t size = 0;
  size_t pos;
  int rc;

  if ((rc = mu_stream_size (stream, &size)))
    return rc;

  switch (whence)
    {
    case SEEK_SET:
      pos = off;
      break;

    case SEEK_CUR:
      pos = off + stream->offset;
      break;

    case SEEK_END:
      pos = size + off;
      break;

    default:
      return EINVAL;
    }

  if (pos > size)
    return EIO;

  stream->offset = pos;
  return 0;
}

int
mu_stream_wait (mu_stream_t stream, int *pflags, struct timeval *tvp)
{
  if (stream == NULL)
    return EINVAL;

  /* Take to acount if we have any buffering.  */
  if ((*pflags) & MU_STREAM_READY_RD)
    {
      if (stream->rbuffer.count > 0)
	{
	  *pflags = 0;
	  *pflags |= MU_STREAM_READY_RD;
	  return 0;
	}
    }

  if (stream->_wait)
    return stream->_wait (stream, pflags, tvp);
  return ENOSYS;
}

int
mu_stream_strerror (mu_stream_t stream, const char **p)
{
  if (stream == NULL)
    return EINVAL;
  if (stream->_strerror)
    return stream->_strerror (stream, p);
  return ENOSYS;
}

static int
refill (mu_stream_t stream, mu_off_t offset)
{
  if (stream->_read)
    {
      int status;
      if (stream->rbuffer.base == NULL)
	{
	  stream->rbuffer.base = calloc (1, stream->rbuffer.bufsiz);
	  if (stream->rbuffer.base == NULL)
	    return ENOMEM;
	}
      stream->rbuffer.ptr = stream->rbuffer.base;
      stream->rbuffer.offset = offset;
      stream->rbuffer.count = 0;
      status = stream->_read (stream, stream->rbuffer.ptr,
			      stream->rbuffer.bufsiz, offset,
			      &stream->rbuffer.count);
      return status;
    }
  return ENOSYS;
}
