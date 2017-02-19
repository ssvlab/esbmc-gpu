/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2008, 2009, 2010 Free Software
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
   MA 02110-1301 USA
*/

#include <iostream>
#include <vector>
#include <mailutils/cpp/mailutils.h>

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace mailutils;

int
main ()
{
  char str[1024];

 again:
  while (!cin.getline (str, sizeof (str)).eof ())
    {
      if (strspn (str, " \t") == strlen (str))
        continue; /* skip empty lines */

      try {
	Url url (str);
	url.parse ();

	cout << "\tscheme <" << url.get_scheme () << ">" << endl;
	cout << "\tuser <"   << url.get_user   () << ">" << endl;

	Secret sec = url.get_secret ();
	cout << "\tpasswd <" << sec.password () << ">" << endl;
	sec.password_unref ();

	cout << "\tauth <"   << url.get_auth   () << ">" << endl;
	cout << "\thost <"   << url.get_host   () << ">" << endl;
	cout << "\tport "    << url.get_port   () << endl;
	cout << "\tpath <"   << url.get_path   () << ">" << endl;

	vector<string> params = url.get_query ();
	for (vector<string>::size_type i = 0; i != params.size (); i++) {
	  cout << "\tquery[" << i << "] <"  << params[i] << ">" << endl;
	}
      }
      catch (Exception& e) {
	cerr << e.method () << ": " << e.what () << endl;
	goto again;
      }
    }
  return 0;
}

