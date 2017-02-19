/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2007, 2010 Free Software Foundation, Inc.

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

/*
 * se[t] [name[=[string]] ...] [name=number ...] [noname ...]
 */

/*
 * NOTE: ask is a synonym for asksub
 */

int
mail_set (int argc, char **argv)
{
  int flags = MOPTF_OVERWRITE | ((strcmp (argv[0], "setq") == 0) ? MOPTF_QUIET : 0);

  if (argc < 2)
    {
      mailvar_print (1);
      return 0;
    }
  else
    {
      int i = 0;

      for (i = 1; i < argc; i++)
	{
	  char *value = strchr (argv[i], '=');
	  if (value)
	    *value++ = 0;
	  
	  if (!strncmp ("no", argv[i], 2) && !value)
	    {
	      mailvar_set (&argv[i][2], NULL, mailvar_type_boolean,
			   flags | MOPTF_UNSET);
	    }
	  else if (value)
	    {
	      int nval;
	      char *p;
	      
	      nval = strtoul (value, &p, 0);
	      if (*p == 0)
		mailvar_set (argv[i], &nval, mailvar_type_number, flags);
	      else
		mailvar_set (argv[i], value, mailvar_type_string, flags);
	    }
	  else
	    {
	      int dummy = 1;
	      mailvar_set (argv[i], &dummy, mailvar_type_boolean, flags);
	    }
	}
      return 0;
    }
  return 1;
}
