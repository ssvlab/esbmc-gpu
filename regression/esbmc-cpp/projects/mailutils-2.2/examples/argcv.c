/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2009, 2010 Free Software Foundation, Inc.

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
#include <mailutils/argcv.h>
#include <mailutils/errno.h>

int
main (int argc, char **argv)
{
  char *delim = "";
  char *comment = "#";
  char buf[512];
  
  while (fgets (buf, sizeof buf, stdin))
    {
      int status, c;
      char **v;
      char *s;

      status = mu_argcv_get (buf, delim, comment, &c, &v);
      if (status)
	{
	  fprintf (stderr, "cannot parse: %s\n", mu_strerror (status));
	  continue;
	}
      status = mu_argcv_string (c, v, &s);
      if (status)
	fprintf (stderr, "cannot create string: %s\n", mu_strerror (status));
      else
	{
	  printf ("%d: %s\n", c, s);
	  free (s);
	}
      mu_argcv_free (c, v);
    } 
  exit (0);
}
