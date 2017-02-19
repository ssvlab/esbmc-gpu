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
#include <cstdlib>
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

int
main (int argc, char **argv)
{
  size_t total = 0;
  char buffer[80];

  if (argc != 3)
    {
      cerr << "usage: " << argv[0] << " from-code to-code" << endl;
      return 1;
    }

  try {
    StdioStream *in = new StdioStream (stdin, 0);
    in->open ();

    FilterIconvStream cvt (*in, (string)argv[1], (string)argv[2], 0,
			   mu_fallback_none);
    cvt.open ();
    delete in;
    
    StdioStream out (stdout, 0);
    out.open ();

    do {
      cvt.read (buffer, sizeof (buffer), total);
      out.sequential_write (buffer, cvt.get_read_count ());
      total += cvt.get_read_count ();
    } while (cvt.get_read_count ());

    out.flush ();
    delete in;
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }

  return 0;
}
