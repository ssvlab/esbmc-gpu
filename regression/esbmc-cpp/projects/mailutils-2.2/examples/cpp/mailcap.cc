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
  string file ("/etc/mailcap");

  try {
    FileStream stream (file, MU_STREAM_READ);
    stream.open ();

    Mailcap mailcap (stream);

    int count = mailcap.entries_count ();

    for (int i = 1; i <= count; i++)
      {
	cout << "entry[" << i << "]\n";

	MailcapEntry entry = mailcap[i];

	/* typefield. */
	cout << "\ttypefield: " << entry.get_typefield () << endl;

	/* view-command. */
	cout << "\tview-command: " << entry.get_viewcommand () << endl;

	/* fields. */
	size_t fields_count = entry.fields_count ();
	for (size_t j = 1; j <= fields_count; j++)
	  {
	    try {
	      cout << "\tfields[" << j << "]: " << entry[j] << endl;
	    }
	    catch (Exception& e) {
	      cerr << e.method () << ": cannot retrieve field "
		   << j << ": " << e.what () << endl;
	    }
	  }
	cout << endl;
      }
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }
  
  return 0;
}

