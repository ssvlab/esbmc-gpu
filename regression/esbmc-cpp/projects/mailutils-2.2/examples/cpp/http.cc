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

/* This is an example program to illustrate the use of stream functions.
   It connects to a remote HTTP server and prints the contents of its
   index page */

#include <iostream>
#include <cstdlib>
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

int
main (int argc, char **argv)
{
  int off = 0;
  string host;

  if (argc == 1)
    host = "www.gnu.org";
  else
    host = argv[1];

  try {
    TcpStream stream (host, 80, MU_STREAM_NONBLOCK);

  connect_again:
    try {
      stream.open ();
    }
    catch (Stream::EAgain) {
      stream.wait (MU_STREAM_READY_WR);
      goto connect_again;
    }

  string path = argc == 3 ? argv[2] : "/";
  string wbuf = "GET " + path + " HTTP/1.0\r\n\r\n";

  write_again:
    try {
      stream << wbuf.substr (off);
    }
    catch (Stream::EAgain) {
      stream.wait (MU_STREAM_READY_WR);
      off += stream.get_write_count ();
      goto write_again;
    }

    if (stream.get_write_count () != wbuf.length ())
      {
	cerr << "stream.get_write_count() != wbuf length" << endl;
	exit (1);
      }

  string rbuf;
  read_again:
    do
      {
	try {
	  stream >> rbuf;
	}
	catch (Stream::EAgain) {
	  stream.wait (MU_STREAM_READY_RD);
	  goto read_again;
	}
	cout << rbuf.substr (0, stream.get_read_count ());
      }
    while (stream.get_read_count ());
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }

  return 0;
}

