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
#include <mailutils/argcv.h>

using namespace std;
using namespace mailutils;

void
usage (int code)
{
  cout << "usage: listop [item..]" << endl;
  exit (code);
}

void
print (List& lst)
{
  for (Iterator itr = lst.begin (); !itr.is_done (); itr++)
    {
      char* text = (char *) *itr;
      cout << text << endl;
    }
}

void
next (Iterator* itr, char *arg)
{
  int skip = arg ? strtoul (arg, NULL, 0) :  1;

  if (skip == 0)
    cout << "next arg?" << endl;
  while (skip--)
    itr->next ();
}

void
del (List& lst, int argc, char **argv)
{
  int rc;
  if (argc == 1)
    {
      cerr << "del arg?" << endl;
      return;
    }

  while (--argc)
    {
      try {
	lst.remove (strdup (*++argv));
      }
      catch (Exception& e) {
	cerr << e.method () << ": " << e.what () << endl;
      }
    }
}

void
add (List& lst, int argc, char **argv)
{
  int rc;
  
  if (argc == 1)
    {
      cerr << "add arg?" << endl;
      return;
    }

  while (--argc)
    {
      try {
	lst.append (strdup (*++argv));
      }
      catch (Exception& e) {
	cerr << e.method () << ": " << e.what () << endl;
      }
    }
}

void
prep (List& lst, int argc, char **argv)
{
  int rc;
  if (argc == 1)
    {
      cerr << "add arg?" << endl;
      return;
    }

  while (--argc)
    {
      try {
	lst.prepend (strdup (*++argv));
      }
      catch (Exception& e) {
	cerr << e.method () << ": " << e.what () << endl;
      }
    }
}

void
repl (List& lst, int argc, char **argv)
{
  int rc;
  if (argc != 3)
    {
      cerr << "repl src dst?" << endl;
      return;
    }

  try {
    lst.replace (argv[1], strdup (argv[2]));
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
  }
}

#define NITR 4

void
iter (int *pnum, int argc, char** argv)
{
  int n;
  if (argc != 2)
    {
      cerr << "iter num?" << endl;
      return;
    }

  n = strtoul (argv[1], NULL, 0);
  if (n < 0 || n >= NITR)
    {
      cerr << "iter [0-3]?" << endl;
      return;
    }
  *pnum = n;
}

void
find (Iterator* itr, char* arg)
{
  char *text;
  if (!arg)
    {
      cerr << "find item?" << endl;
      return;
    }

  itr->current ((void**) &text);
  for (itr->first (); !itr->is_done (); itr->next ())
    {
      char *item;

      itr->current ((void**) &item);
      if (!strcmp (arg, item))
	return;
    }

  cerr << arg << " not in list" << endl;

  for (itr->first (); !itr->is_done (); itr->next ())
    {
      char *item;

      itr->current ((void**) &item);
      if (!strcmp (text, item))
	return;
    }
}

void
help ()
{
  cout << "next [count]\n";
  cout << "first\n";
  cout << "find item\n";
  cout << "del item [item...]\n";
  cout << "add item [item...]\n";
  cout << "prep item [item...]\n";
  cout << "repl old_item new_item\n";
  cout << "print\n";
  cout << "quit\n";
  cout << "iter num\n";
  cout << "help\n";
  cout << "NUMBER\n";
}

void
shell (List& lst)
{
  int rc;
  int num = 0;
  Iterator* itr[NITR];

  for (num = 0; num < NITR; num++)
    {
      itr[num] = new Iterator (lst);
      itr[num]->first ();
    }

  num = 0;
  while (1)
    {
      char *text;
      char buf[80];
      int argc;
      char **argv;
      
      try {
	itr[num]->current ((void**) &text);
      }
      catch (Exception& e) {
	cerr << e.method () << ": " << e.what () << endl;
      }

      cout << num << ":(" << (text ? text : "NULL") << ")> ";
      if (cin.getline (buf, sizeof (buf)).eof ())
	return;

      rc = mu_argcv_get (buf, "", "#", &argc, &argv);
      if (rc)
	cerr << "mu_argcv_get: " << rc << endl;

      if (argc > 0)
	{
	  string cmd (argv[0]);

	  if (cmd == "next")
	    next (itr[num], argv[1]);
	  else if (cmd == "first")
	    itr[num]->first ();
	  else if (cmd == "del")
	    del (lst, argc, argv);
	  else if (cmd == "add")
	    add (lst, argc, argv);
	  else if (cmd == "prep")
	    prep (lst, argc, argv);
	  else if (cmd == "repl")
	    repl (lst, argc, argv);
	  else if (cmd == "print")
	    print (lst);
	  else if (cmd == "quit")
	    return;
	  else if (cmd == "iter")
	    iter (&num, argc, argv);
	  else if (cmd == "find")
	    find (itr[num], argv[1]);
	  else if (cmd == "help")
	    help ();
	  else if (argc == 1)
	    {
	      char* p;
	      size_t n = strtoul (argv[0], &p, 0);
	      if (*p != 0)
		cerr << "?" << endl;
	      else
		{
		  try {
		    text = (char*) lst[n];
		  }
		  catch (Exception& e) {
		    cerr << e.method () << ": " << e.what () << endl;
		  }

		  // else
		  cout << text << endl;
		}
	    }
	  else
	    cerr << "?" << endl;
	}
      mu_argcv_free (argc, argv);
    }
}

static int
string_comp (const void* item, const void* value)
{
  return strcmp ((const char*) item, (const char*) value);
}

int
main (int argc, char **argv)
{
  int rc;

  while ((rc = getopt (argc, argv, "h")) != EOF)
    switch (rc)
      {
      case 'h':
	usage (0);
	
      default:
	usage (1);
      }

  argc -= optind;
  argv += optind;

  try {
    List lst;
    lst.set_comparator (string_comp);
    
    while (argc--)
      {
	lst.append (*argv++);
      }

    shell (lst);
  }
  catch (Exception& e) {
    cerr << e.method () << ": " << e.what () << endl;
  }

  return 0;
}

