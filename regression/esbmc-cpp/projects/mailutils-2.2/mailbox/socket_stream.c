/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2010
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <signal.h>
#include <errno.h>

#include <mailutils/stream.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>

struct _socket_stream
{
  mu_stream_t fstream;
  char *filename;
  int ec; /* Last error code if fstream == NULL */
};

static void
_s_destroy (mu_stream_t stream)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);

  if (s->filename)
    free (s->filename);
  mu_stream_destroy (&s->fstream, mu_stream_get_owner (s->fstream));
  free (s);
}

static int
_s_read (mu_stream_t stream, char *optr, size_t osize,
	 mu_off_t offset, size_t *nbytes)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_read (s->fstream, optr, osize, offset, nbytes);
}

static int
_s_readline (mu_stream_t stream, char *optr, size_t osize,
	     mu_off_t offset, size_t *nbytes)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_readline (s->fstream, optr, osize, offset, nbytes);
}

static int
_s_write (mu_stream_t stream, const char *iptr, size_t isize,
	  mu_off_t offset, size_t *nbytes)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_write (s->fstream, iptr, isize, offset, nbytes);
}

static int
_s_open (mu_stream_t stream)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  int fd, rc;
  FILE *fp;
  struct sockaddr_un addr;
  char *fstr;
  int flags;
  
  if (!s)
    return EINVAL;
  
  fd = socket (PF_UNIX, SOCK_STREAM, 0);
  if (fd < 0)
    return errno;
  
  memset (&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  strncpy (addr.sun_path, s->filename, sizeof addr.sun_path - 1);
  addr.sun_path[sizeof addr.sun_path - 1] = 0;
  if (connect (fd, (struct sockaddr *) &addr, sizeof(addr)))
    {
      close (fd);
      return errno;
    }

  mu_stream_get_flags(stream, &flags);
  if (flags & MU_STREAM_WRITE)
    fstr = "w";
  else if (flags & MU_STREAM_RDWR)
    fstr = "w+";
  else if (flags & MU_STREAM_READ)
    fstr = "r";
  else
    fstr = "w+";
  
  fp = fdopen (fd, fstr);
  if (!fp)
    {
      close (fd);
      return errno;
    }
      
  rc = mu_stdio_stream_create (&s->fstream, fp, flags);
  if (rc)
    {
      fclose (fp);
      return rc;
    }
  
  rc = mu_stream_open (s->fstream);
  if (rc)
    {
      mu_stream_destroy (&s->fstream, mu_stream_get_owner (s->fstream));
      fclose (fp);
    }
  return rc;
}

static int
_s_close (mu_stream_t stream)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_close (s->fstream);
}

static int
_s_flush (mu_stream_t stream)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_flush (s->fstream);
}

int
_s_wait (mu_stream_t stream, int *pflags, struct timeval *tvp)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_wait (s->fstream, pflags, tvp);
}

int
_s_strerror (mu_stream_t stream, const char **pstr)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_strerror (s->fstream, pstr);
}

static int
_s_get_transport2 (mu_stream_t stream,
		   mu_transport_t *pin, mu_transport_t *pout)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  return mu_stream_get_transport2 (s->fstream, pin, pout);
}

int
_s_shutdown (mu_stream_t stream, int how)
{
  struct _socket_stream *s = mu_stream_get_owner (stream);
  int flag;
  mu_transport_t trans;

  if (s->fstream == NULL)
    return EINVAL;

  mu_stream_get_transport(s->fstream, &trans);
  switch (how)
    {
    case MU_STREAM_READ:
      flag = SHUT_RD;
      break;
      
    case MU_STREAM_WRITE:
      flag = SHUT_WR;
    }

  if (shutdown ((int) trans, flag))
    return errno;
  return 0;
}

int
mu_socket_stream_create (mu_stream_t *stream, const char *filename, int flags)
{
  struct _socket_stream *s;
  int rc;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  s = calloc (1, sizeof (struct _socket_stream));
  if (s == NULL)
    return ENOMEM;

  if ((s->filename = strdup (filename)) == NULL)
    {
      free (s);
      return ENOMEM;
    }

  rc = mu_stream_create (stream, flags | MU_STREAM_NO_CHECK, s);
  if (rc)
    {
      free (s);
      free (s->filename);
      return rc;
    }

  mu_stream_set_open (*stream, _s_open, s);
  mu_stream_set_close (*stream, _s_close, s);
  mu_stream_set_get_transport2 (*stream, _s_get_transport2, s);
  mu_stream_set_read (*stream, _s_read, s);
  mu_stream_set_readline (*stream, _s_readline, s);
  mu_stream_set_write (*stream, _s_write, s);
  mu_stream_set_flush (*stream, _s_flush, s);
  mu_stream_set_destroy (*stream, _s_destroy, s);
  mu_stream_set_strerror (*stream, _s_strerror, s);
  mu_stream_set_wait (*stream, _s_wait, s);
  mu_stream_set_shutdown (*stream, _s_shutdown, s);
  
  return 0;
}




