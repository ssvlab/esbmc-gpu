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

int main (int argc, char* argv[])
{
  register_local_mbox_formats ();

  try {
    MailboxDefault mbox ((argc > 1) ? argv[1] : "");
    mbox.open ();

    size_t total = mbox.messages_count ();
    cout << "Total: " << total << endl;

    for (int msgno = 1; msgno <= total; msgno++)
    {
      Message msg = mbox[msgno];
      Header hdr = msg.get_header ();
      cout << hdr[MU_HEADER_FROM] << " "
	   << hdr.get_value (MU_HEADER_SUBJECT, "(NO SUBJECT)") << endl;
    }

    mbox.close ();
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
    exit (1);
  }

  return 0;
}

