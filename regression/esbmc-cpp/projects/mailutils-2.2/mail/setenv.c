/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2007, 2010 Free Software
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

#include "mail.h"

#if !HAVE_DECL_ENVIRON
extern char **environ;
#endif

int
mail_setenv (int argc, char **argv)
{
  if (argc == 1)
    {
      char **p;
      
      for (p = environ; *p; p++)
	printf ("%s\n", *p);
    }
  else
    {
      int i;

      /* Note: POSIX requires that the argument to putenv become part
	 of the environment itself, hence the memory allocation. */

      for (i = 1; i < argc; i++)
	{
	  char *value = strchr (argv[i], '=');
	  if (value)
	    *value++ = 0;
	  setenv (argv[i], value ? value : "", 1);
	}
    }
  return 0;
}

