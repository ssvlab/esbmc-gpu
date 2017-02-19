/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include <mailutils/mailutils.h>

int
main (int argc, char **argv)
{
  int rc;
  mu_stream_t in, out;
  mu_stream_t cvt;
  size_t total = 0, size;
  char buffer[80];
  
  if (argc != 3)
    {
      fprintf (stderr, "usage: %s from-code to-code\n", argv[0]);
      return 1;
    }

  MU_ASSERT (mu_stdio_stream_create (&in, stdin, 0));
  MU_ASSERT (mu_stream_open (in));
  MU_ASSERT (mu_filter_iconv_create (&cvt, in, argv[1], argv[2], 
                                     0, mu_fallback_none));
  MU_ASSERT (mu_stream_open (cvt));
  
  MU_ASSERT (mu_stdio_stream_create (&out, stdout, 0));
  MU_ASSERT (mu_stream_open (out));

  while ((rc = mu_stream_read (cvt, buffer, sizeof (buffer), total, &size)) == 0
	 && size > 0)
    {
      mu_stream_sequential_write (out, buffer, size);
      total += size;
    }
  mu_stream_flush (out);
  if (rc)
    {
      const char *p;
      mu_stream_strerror (cvt, &p);
      fprintf (stderr, "error: %s / %s\n", mu_strerror (rc), p);
    }
  return 0;
}
