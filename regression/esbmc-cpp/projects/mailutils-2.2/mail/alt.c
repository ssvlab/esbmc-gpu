/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2007, 2009, 2010 Free Software
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
#include <pwd.h>

static mu_list_t alternate_names = NULL;
static char *my_email;
static char *my_name;

/*
 * alt[ernates] name...
 */

int
mail_alt (int argc, char **argv)
{
  if (argc == 1)
    {
      if (alternate_names)
	{
	  util_slist_print (alternate_names, 0);
	  fprintf (ofile, "\n");
	}
    }
  else
    {
      util_slist_destroy (&alternate_names);
      while (--argc)
	util_slist_add (&alternate_names, *++argv);
    }
  return 0;
}

char *
mail_whoami ()
{
  return my_name;
}

void
mail_set_my_name (char *name)
{
  if (!name)
    {
      struct passwd *pw = getpwuid (getuid ());
      if (!pw)
	{
	  util_error (_("Cannot determine my username"));
	  exit (1);
	}
      name = pw->pw_name;
    }
  my_name = strdup (name);
  my_email = mu_get_user_email (name);
  if (!my_email)
    {
      util_error(_("Cannot determine my email address: %s"),
		 mu_strerror (errno));
      exit (1);
    }
}
   
int
mail_is_my_name (const char *name)
{
  if (strchr(name, '@') == NULL && mu_c_strcasecmp (name, my_name) == 0)
    return 1;
  if (mu_c_strcasecmp (name, my_email) == 0)
    return 1;
  return util_slist_lookup (alternate_names, name);
}
