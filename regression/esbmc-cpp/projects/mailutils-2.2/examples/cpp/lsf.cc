/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include <iostream>
#include <cstdlib>
#include <mailutils/cpp/mailutils.h>

using namespace std;
using namespace mailutils;

static int
enumfun (mu_folder_t folder, struct mu_list_response *resp, void *data)
{
  printf ("%c%c %c %4d %s\n",
	  (resp->type & MU_FOLDER_ATTRIBUTE_DIRECTORY) ? 'd' : '-',
	  (resp->type & MU_FOLDER_ATTRIBUTE_FILE) ? 'f' : '-',
	  resp->separator,
	  resp->level,
	  resp->name);
  return 0;
}

int
ls_folders (const string& fname, const string& ref, void* pattern, int level)
{
  try {
    Folder folder (fname);
    folder.open ();

    List list = folder.enumerate (ref, pattern, 0, level, enumfun, NULL);
    cout << "Number of folders: " << list.count () << endl;

    folder.close ();
  }
  catch (Exception& e)
  {
    if (e.status () == MU_ERR_NOENT) {
      cout << "No folders matching " << ref << " " << pattern
	   << " in " << fname << endl;
    }
    else {
      cerr << e.method () << ": " << e.what () << endl;
      return 1;
    }
  }
  return 0;
}

int
main (int argc, char *argv[])
{
  string folder;
  string ref;
  const char *pattern = "*";
  int level = 0;

  switch (argc)
    {
    case 5:
      level = atoi (argv[4]);
    case 4:
      pattern = argv[3];
    case 3:
      ref = argv[2];
    case 2:
      folder = argv[1];
      break;
    default:
      error ("usage: lsf folder [ref] [pattern] [recursion-level]\n");
      return 1;
    }

  register_all_mbox_formats ();

  return ls_folders (folder, ref, (char *)pattern, level);
}

