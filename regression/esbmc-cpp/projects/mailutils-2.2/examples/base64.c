/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <mailutils/mailutils.h>

#define ISPRINT(c) ((c)>=' '&&(c)<127) 

int
main (int argc, char * argv [])
{
  mu_stream_t in, out, flt;
  unsigned char buffer;
  int c, verbose = 0;
  int printable = 0;
  size_t size, total = 0;
  int mode = MU_FILTER_ENCODE;
  char *input = NULL, *output = NULL;
  char *encoding = "base64";
    
  while ((c = getopt (argc, argv, "deE:hi:o:pv")) != EOF)
    switch (c)
      {
      case 'i':
	input = optarg;
	break;

      case 'o':
	output = optarg;
	break;
	
      case 'd':
	mode = MU_FILTER_DECODE;
	break;

      case 'E':
	encoding = optarg;
	break;
	
      case 'e':
	mode = MU_FILTER_ENCODE;
	break;

      case 'p':
 	printable = 1;
	break;
	
      case 'v':
	verbose = 1;
	break;

      case 'h':
	printf ("usage: base64 [-vpde][-E encoding][-i infile][-o outfile]\n");
	exit (0);
	  
      default:
	exit (1);
      }

  if (input)
    MU_ASSERT (mu_file_stream_create (&in, input, MU_STREAM_READ));
  else
    MU_ASSERT (mu_stdio_stream_create (&in, stdin, 0));
  MU_ASSERT (mu_filter_create (&flt, in, encoding, mode, MU_STREAM_READ));
  MU_ASSERT (mu_stream_open (in));

  if (output)
    MU_ASSERT (mu_file_stream_create (&out, output, 
                                      MU_STREAM_WRITE|MU_STREAM_CREAT));
  else
    MU_ASSERT (mu_stdio_stream_create (&out, stdout, 0));
  MU_ASSERT (mu_stream_open (out));
  
  while (mu_stream_read (flt, (char*) &buffer,
			 sizeof (buffer), total, &size) == 0
	 && size > 0)
    {
      if (printable && !ISPRINT (buffer))
	{
	  char outbuf[24];
	  sprintf (outbuf, "\\%03o", (unsigned int) buffer);
	  mu_stream_sequential_write (out, outbuf, strlen (outbuf));
	} 
      else
	mu_stream_sequential_write (out, (char*) &buffer, size);
      total += size;
    }

  mu_stream_sequential_write (out, "\n", 1);
  mu_stream_close (in);
  mu_stream_close (out);
  if (verbose)
    fprintf (stderr, "total: %lu bytes\n", (unsigned long) total);
  return 0;
}
