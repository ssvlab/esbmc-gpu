/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <mailutils/stream.h>
#include <mailutils/mutil.h>
#include <mailutils/errno.h>
#include <body0.h>

#define BODY_MODIFIED 0x10000

static int _body_flush    (mu_stream_t);
static int _body_get_transport2 (mu_stream_t, mu_transport_t *, mu_transport_t *);
static int _body_read     (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int _body_readline (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int _body_truncate (mu_stream_t, mu_off_t);
static int _body_size     (mu_stream_t, mu_off_t *);
static int _body_write    (mu_stream_t, const char *, size_t, mu_off_t, size_t *);

/* Our own defaults for the body.  */
static int _body_get_size   (mu_body_t, size_t *);
static int _body_get_lines  (mu_body_t, size_t *);
static int _body_get_size0  (mu_stream_t, size_t *);
static int _body_get_lines0 (mu_stream_t, size_t *);

int
mu_body_create (mu_body_t *pbody, void *owner)
{
  mu_body_t body;

  if (pbody == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (owner == NULL)
    return EINVAL;

  body = calloc (1, sizeof (*body));
  if (body == NULL)
    return ENOMEM;

  body->owner = owner;
  *pbody = body;
  return 0;
}

void
mu_body_destroy (mu_body_t *pbody, void *owner)
{
  if (pbody && *pbody)
    {
      mu_body_t body = *pbody;
      if (body->owner == owner)
	{
	  if (body->filename)
	    {
	      /* FIXME: should we do this?  */
	      remove (body->filename);
	      free (body->filename);
	    }

	  if (body->stream)
	    mu_stream_destroy (&(body->stream), body);

	  if (body->fstream)
	    {
	      mu_stream_close (body->fstream);
	      mu_stream_destroy (&(body->fstream), NULL);
	    }

	  free (body);
	}
      *pbody = NULL;
    }
}

void *
mu_body_get_owner (mu_body_t body)
{
  return (body) ? body->owner : NULL;
}

/* FIXME: not implemented.  */
int
mu_body_is_modified (mu_body_t body)
{
  return (body) ? (body->flags & BODY_MODIFIED) : 0;
}

/* FIXME: not implemented.  */
int
mu_body_clear_modified (mu_body_t body)
{
  if (body)
    body->flags &= ~BODY_MODIFIED;
  return 0;
}

int
mu_body_get_filename (mu_body_t body, char *filename, size_t len, size_t *pn)
{
  int n = 0;
  if (body == NULL)
    return EINVAL;
  if (body->filename)
    {
      n = strlen (body->filename);
      if (filename && len > 0)
	{
	  len--; /* Space for the null.  */
	  strncpy (filename, body->filename, len)[len] = '\0';
	}
    }
  if (pn)
    *pn = n;
  return 0;
}

int
mu_body_get_stream (mu_body_t body, mu_stream_t *pstream)
{
  if (body == NULL)
    return EINVAL;
  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (body->stream == NULL)
    {
      int status = mu_stream_create (&body->stream, MU_STREAM_RDWR, body);
      if (status != 0)
	return status;
      /* Create the temporary file.  */
      body->filename = mu_tempname (NULL);
      status = mu_file_stream_create (&body->fstream, 
                                   body->filename, MU_STREAM_RDWR);
      if (status != 0)
	return status;
      status = mu_stream_open (body->fstream);
      if (status != 0)
	return status;
      mu_stream_set_get_transport2 (body->stream, _body_get_transport2, body);
      mu_stream_set_read (body->stream, _body_read, body);
      mu_stream_set_readline (body->stream, _body_readline, body);
      mu_stream_set_write (body->stream, _body_write, body);
      mu_stream_set_truncate (body->stream, _body_truncate, body);
      mu_stream_set_size (body->stream, _body_size, body);
      mu_stream_set_flush (body->stream, _body_flush, body);
      /* Override the defaults.  */
      body->_lines = _body_get_lines;
      body->_size = _body_get_size;
    }
  *pstream = body->stream;
  return 0;
}

int
mu_body_set_stream (mu_body_t body, mu_stream_t stream, void *owner)
{
  if (body == NULL)
   return EINVAL;
  if (body->owner != owner)
    return EACCES;
  /* make sure we destroy the old one if it is own by the body */
  mu_stream_destroy (&(body->stream), body);
  body->stream = stream;
  body->flags |= BODY_MODIFIED;
  return 0;
}

int
mu_body_set_lines (mu_body_t body, int (*_lines) (mu_body_t, size_t *), void *owner)
{
  if (body == NULL)
    return EINVAL;
  if (body->owner != owner)
    return EACCES;
  body->_lines = _lines;
  return 0;
}

int
mu_body_lines (mu_body_t body, size_t *plines)
{
  if (body == NULL)
    return EINVAL;
  if (body->_lines)
    return body->_lines (body, plines);
  /* Fall on the stream.  */
  if (body->stream)
    return _body_get_lines0 (body->stream, plines);
  if (plines)
    *plines = 0;
  return 0;
}

int
mu_body_size (mu_body_t body, size_t *psize)
{
  if (body == NULL)
    return EINVAL;
  if (body->_size)
    return body->_size (body, psize);
  /* Fall on the stream.  */
  if (body->stream)
    return _body_get_size0 (body->stream, psize);
  if (psize)
    *psize = 0;
  return 0;
}

int
mu_body_set_size (mu_body_t body, int (*_size)(mu_body_t, size_t*) , void *owner)
{
  if (body == NULL)
    return EINVAL;
  if (body->owner != owner)
    return EACCES;
  body->_size = _size;
  return 0;
}

/* Stub function for the body stream.  */

static int
_body_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_get_transport2 (body->fstream, pin, pout);
}

static int
_body_read (mu_stream_t stream,  char *buffer, size_t n, mu_off_t off, size_t *pn)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_read (body->fstream, buffer, n, off, pn);
}

static int
_body_readline (mu_stream_t stream, char *buffer, size_t n, mu_off_t off, size_t *pn)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_readline (body->fstream, buffer, n, off, pn);
}

static int
_body_write (mu_stream_t stream, const char *buf, size_t n, mu_off_t off, size_t *pn)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_write (body->fstream, buf, n, off, pn);
}

static int
_body_truncate (mu_stream_t stream, mu_off_t n)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_truncate (body->fstream, n);
}

static int
_body_size (mu_stream_t stream, mu_off_t *size)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_size (body->fstream, size);
}

static int
_body_flush (mu_stream_t stream)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mu_stream_flush (body->fstream);
}

/* Default function for the body.  */
static int
_body_get_lines (mu_body_t body, size_t *plines)
{
  return _body_get_lines0 (body->fstream, plines);
}

static int
_body_get_size (mu_body_t body, size_t *psize)
{
  return _body_get_size0 (body->fstream, psize);
}

static int
_body_get_size0 (mu_stream_t stream, size_t *psize)
{
  mu_off_t off = 0;
  int status = mu_stream_size (stream, &off);
  if (psize)
    *psize = off;
  return status;
}

static int
_body_get_lines0 (mu_stream_t stream, size_t *plines)
{
  int status =  mu_stream_flush (stream);
  size_t lines = 0;
  if (status == 0)
    {
      char buf[128];
      size_t n = 0;
      mu_off_t off = 0;
      while ((status = mu_stream_readline (stream, buf, sizeof buf,
					   off, &n)) == 0 && n > 0)
	{
	  if (buf[n - 1] == '\n')
	    lines++;
	  off += n;
	}
    }
  if (plines)
    *plines = lines;
  return status;
}


