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
#include <cstring>
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

static int
parse (const char *str)
{
  set_user_email_domain ("localhost");

  try {
    Address address (str);
    size_t count = address.get_count ();
    cout << address << " => count " << count << endl;

    for (size_t no = 1; no <= count; no++)
      {
	bool isgroup = address.is_group (no);
	cout << no << " ";
	
	if (isgroup)
	  cout << "group <" << address.get_personal (no) << ">" << endl;
	else
	  cout << "email <" << address.get_email (no) << ">" << endl;
	
	if (!isgroup)
	  cout << "   personal <" << address.get_personal (no) << ">" << endl;
	
	cout << "   comments <" << address.get_comments (no) << ">" << endl;
	cout << "   local-part <" << address.get_local_part (no) << ">"
	     << " domain <"  << address.get_domain (no) << ">" << endl;
	cout << "   route <" << address.get_route (no) << ">" << endl;
      }
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
  }

  cout << endl;
  return 0;
}

static int
parseinput (void)
{
  char buf[BUFSIZ];

  while (!cin.getline (buf, sizeof (buf)).eof ())
    {
      parse (buf);
    }

  return 0;
}

int
main (int argc, const char *argv[])
{
  argc = 1;

  if (!argv[argc])
    return parseinput ();

  for (; argv[argc]; argc++)
    {
      if (strcmp (argv[argc], "-") == 0)
	parseinput ();
      else
	parse (argv[argc]);
    }

  return 0;
}

