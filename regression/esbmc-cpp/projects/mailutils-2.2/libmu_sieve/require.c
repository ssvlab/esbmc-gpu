/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  
#include <string.h>  
#include <sieve-priv.h>

void
mu_sieve_require (mu_list_t slist)
{
  int status;
  mu_iterator_t itr;
  
  status = mu_list_get_iterator (slist, &itr);
  if (status)
    {
      mu_sv_compile_error (&mu_sieve_locus,
			   _("cannot create iterator: %s"),
			   mu_strerror (status));
      return;
    }

  for (mu_iterator_first (itr);
       !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      char *name;
      int (*reqfn) (mu_sieve_machine_t mach, const char *name) = NULL;
      const char *text = NULL;
      
      mu_iterator_current (itr, (void **)&name);

      if (strncmp (name, "comparator-", 11) == 0)
	{
	  name += 11;
	  reqfn = mu_sieve_require_comparator;
	  text = _("required comparator");
	}
      else if (strncmp (name, "test-", 5)  == 0) /* GNU extension */
	{
	  name += 5;
	  reqfn = mu_sieve_require_test;
	  text = _("required test");
	}
      else if (strcmp (name, "relational") == 0) /* RFC 3431 */
	{
	  reqfn = mu_sieve_require_relational;
	  text = "";
	}
      else
	{
	  reqfn = mu_sieve_require_action;
	  text = _("required action");
	}

      if (reqfn (mu_sieve_machine, name))
	{
	  mu_sv_compile_error (&mu_sieve_locus,
			       _("source for the %s %s is not available"),
			       text,
			       name);
	}
    }
  mu_iterator_destroy (&itr);
}
     
