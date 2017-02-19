/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2007, 2010 Free Software
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

/* This is an example program to illustrate the use of stream functions.
   It connects to a remote HTTP server and prints the contents of its
   index page */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/time.h>

#include <mailutils/mailutils.h>

char wbuf[1024];
char rbuf[1024];

size_t io_timeout = 3;
size_t io_attempts = 3;

int
http_stream_wait (mu_stream_t stream, int flags, size_t *attempt)
{
  int rc;
  int oflags = flags;
  struct timeval tv;

  while (*attempt < io_attempts)
    {
      tv.tv_sec = io_timeout;
      tv.tv_usec = 0;
      rc = mu_stream_wait (stream, &oflags, &tv);
      switch (rc) {
      case 0:
	if (flags & oflags)
	  return 0;
	/* FALLTHROUGH */
      case EAGAIN:
      case EINPROGRESS:
	++*attempt;
	continue;
	
      default:
	return rc;
      }
    }
  return ETIMEDOUT;
}

int
main (int argc, char **argv)
{
  int ret, off = 0;
  mu_stream_t stream;
  size_t nb, size;
  size_t attempt;
  char *url = "www.gnu.org";

  if (argc > 3)
    {
      fprintf (stderr, "usage: %s [hostname [url]]\n", argv[0]);
      exit (1);
    }
  
  if (argc > 1)
    url = argv[1];
  
  snprintf (wbuf, sizeof wbuf, "GET %s HTTP/1.0\r\n\r\n",
	    argc == 3 ? argv[2] : "/");

  ret = mu_tcp_stream_create (&stream, url, 80, MU_STREAM_NONBLOCK);
  if (ret != 0)
    {
      mu_error ("mu_tcp_stream_create: %s", mu_strerror (ret));
      exit (EXIT_FAILURE);
    }

  for (attempt = 0; (ret = mu_stream_open (stream)); )
    {
      if ((ret == EAGAIN || ret == EINPROGRESS) && attempt < io_attempts)
	{
	  ret = http_stream_wait(stream, MU_STREAM_READY_WR, &attempt);
	  if (ret == 0)
	    continue;
	}
      mu_error ("mu_stream_open: %s", mu_strerror (ret));
      exit (EXIT_FAILURE);
    }

  for (attempt = 0, size = strlen (wbuf); size > 0; )
    {
      ret = mu_stream_write (stream, wbuf + off, strlen (wbuf), 0, &nb);
      if (ret == 0)
	{
	  if (nb == 0)
	    {
	      mu_error("mu_stream_write: wrote 0 bytes");
	      exit (EXIT_FAILURE);
	    }
	  off += nb;
	  size -= nb;
	}
      else if (ret == EAGAIN)
        {
	  if (attempt < io_attempts)
	    {
	      ret = http_stream_wait (stream, MU_STREAM_READY_WR, &attempt);
	      if (ret)
		{
		  mu_error ("http_wait failed: %s", mu_strerror (ret));
		  return -1;
		}
	      continue;
	    }
	  else
	    {
	      mu_error ("mu_stream_write timed out");
	      exit (EXIT_FAILURE);
	    }
	}
      else
	{
	  mu_error ("mu_stream_write: %s", mu_strerror (ret));
	  exit (EXIT_FAILURE);
	}
    }

  attempt = 0;
  for (;;)
    {
      ret = mu_stream_read (stream, rbuf, sizeof (rbuf), 0, &nb);
      if (ret == 0)
	{
	  if (nb == 0)
	    break;
	  write (1, rbuf, nb);
	}
      else if (ret == EAGAIN)
	{
	  if (attempt < io_attempts)
	    {
	      ret = http_stream_wait (stream, MU_STREAM_READY_RD, &attempt);
	      if (ret)
		{
		  mu_error ("http_stream_wait failed: %s", mu_strerror (ret));
		  exit (EXIT_FAILURE);
		}
	    }
	  else
	    {
	      mu_error ("mu_stream_read: %s", mu_strerror (ret));
	      exit (EXIT_FAILURE);
	    }
	}
    }

  ret = mu_stream_close (stream);
  if (ret != 0)
    {
      mu_error ("mu_stream_close: %s", mu_strerror (ret));
      exit (EXIT_FAILURE);
    }

  mu_stream_destroy (&stream, NULL);
  exit (EXIT_SUCCESS);
}
