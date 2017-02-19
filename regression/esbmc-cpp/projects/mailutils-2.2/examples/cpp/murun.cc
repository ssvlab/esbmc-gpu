/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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
   MA 02110-1301 USA
*/

#include <iostream>
#include <mailutils/cpp/mailutils.h>
#include <mailutils/argcv.h>

using namespace std;
using namespace mailutils;

static char *progname;

static void
read_and_print (Stream *in, Stream& out)
{
  char buffer[128];
  
  in->sequential_readline (buffer, sizeof (buffer));
  while (in->get_read_count ())
    {
      out.sequential_write (buffer, in->get_read_count ());
      in->sequential_readline (buffer, sizeof (buffer));
    }
}

Stream *
create_filter (bool read_stdin, char *cmdline, int flags)
{
  try {
    if (read_stdin)
      {
	StdioStream *in = new StdioStream (stdin, 0);
	in->open ();
	FilterProgStream *stream = new FilterProgStream (cmdline, in);
	stream->open ();
	return stream;
      }
    else
      {
	ProgStream *stream = new ProgStream (cmdline, flags);
	stream->open ();
	return stream;
      }
  }
  catch (Exception& e) {
    cerr << progname << ": cannot create program filter stream: "
	 << e.method () << ": " << e.what () << endl;
    exit (1);
  }
}

int
main (int argc, char *argv[])
{
  int i = 1;
  int read_stdin = 0;
  int flags = MU_STREAM_READ;
  char *cmdline;
  Stream *stream, out;

  progname = argv[0];
  
  if (argc > 1 && strcmp (argv[i], "--stdin") == 0)
    {
      read_stdin = 1;
      flags |= MU_STREAM_WRITE;
      i++;
    }

  if (i == argc)
    {
      cerr << "Usage: " << argv[0] << " [--stdin] progname [args]" << endl;
      exit (1);
    }

  mu_argcv_string (argc - i, &argv[i], &cmdline);

  stream = create_filter (read_stdin, cmdline, flags);

  try {
    StdioStream out (stdout, 0);
    out.open ();

    read_and_print (stream, out);

    delete stream;
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }

  return 0;
}

