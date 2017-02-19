/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <mailutils/address.h>
#include <mailutils/errno.h>
#include <mailutils/kwd.h>
#include <mailutils/mutil.h>

#define EPARSE MU_ERR_NOENT

struct mu_address hint;
int hflags;

static int
parse (const char *str)
{
  size_t no = 0;
  size_t pcount = 0;
  int status;
  const char *buf;
  mu_address_t address = NULL;

  status = mu_address_create_hint (&address, str, &hint, hflags);
  mu_address_get_count (address, &pcount);

  if (status)
    {
      printf ("%s=> error %s\n\n", str, mu_errname (status));
      return 0;
    }
  else
    {
      printf ("%s=> pcount %lu\n", str, (unsigned long) pcount);
    }

  for (no = 1; no <= pcount; no++)
    {
      int isgroup;

      mu_address_is_group (address, no, &isgroup);
      printf ("%lu ", (unsigned long) no);

      if (isgroup)
        {
          mu_address_sget_personal (address, no, &buf);
          printf ("group <%s>\n", buf);
        }
      else
        {
          mu_address_sget_email (address, no, &buf);
          printf ("email <%s>\n", buf);
        }

      if (mu_address_sget_personal (address, no, &buf) == 0 && buf && !isgroup)
        printf ("   personal <%s>\n", buf);

      if (mu_address_sget_comments (address, no, &buf) == 0 && buf)
        printf ("   comments <%s>\n", buf);

      if (mu_address_sget_local_part (address, no, &buf) == 0 && buf)
        {
          printf ("   local-part <%s>", buf);

          if (mu_address_sget_domain (address, no, &buf) == 0 && buf)
            printf (" domain <%s>", buf);

          printf ("\n");
        }

      if (mu_address_sget_route (address, no, &buf) == 0 && buf)
        printf ("   route <%s>\n", buf);
    }
  mu_address_destroy (&address);

  printf ("\n");
  return 0;
}

struct mu_kwd hintnames[] = {
  { "comments", MU_ADDR_HINT_COMMENTS },
  { "personal", MU_ADDR_HINT_PERSONAL },
  { "email", MU_ADDR_HINT_EMAIL },
  { "local", MU_ADDR_HINT_LOCAL },
  { "domain", MU_ADDR_HINT_DOMAIN },
  { "route", MU_ADDR_HINT_ROUTE },
  { NULL }
};

static char **
addr_fieldptr_by_mask (mu_address_t addr, int mask)
{
  switch (mask)						
    {
    case MU_ADDR_HINT_ADDR:
      return &addr->addr;
	  
    case MU_ADDR_HINT_COMMENTS:				
      return &addr->comments;					
	  
    case MU_ADDR_HINT_PERSONAL:				
      return &addr->personal;					

    case MU_ADDR_HINT_EMAIL:
      return &addr->email;

    case MU_ADDR_HINT_LOCAL:
      return &addr->local_part;
      
    case MU_ADDR_HINT_DOMAIN:				
      return &addr->domain;					

    case MU_ADDR_HINT_ROUTE:
      return &addr->route;
    }
  return NULL;
}							

void
sethint (char *str)
{
  int mask;
  char *p = strchr (str, '=');

  if (!p)
    {
      printf ("%s=> bad assignment\n\n", str);
      return;
    }
  *p++ = 0;
  if (mu_kwd_xlat_name (hintnames, str, &mask) == 0)
    {
      char **fptr = addr_fieldptr_by_mask (&hint, mask);

      if (*p == 0)
	hflags &= ~mask;
      else
	{
	  *fptr = strdup (p);
	  hflags |= mask;
	}
    }
  else
    printf ("%s=> unknown hint name\n\n", str);
}
	
static int
parseinput (void)
{
  char buf[BUFSIZ];

  while (fgets (buf, sizeof (buf), stdin) != 0)
    {
      buf[strlen (buf) - 1] = 0;
      if (buf[0] == '\\')
	sethint (buf + 1);
      else
	parse (buf);
    }

  return 0;
}

int
main (int argc, char *argv[])
{
  int i;
  
  hint.domain = "localhost";
  hflags = MU_ADDR_HINT_DOMAIN;
  
  if (argc == 1)
    return parseinput ();
  
  for (i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "-") == 0)
	parseinput ();
      else if (strncmp (argv[i], "-v", 2) == 0)
	sethint (argv[i] + 2);
      else
	parse (argv[i]);
    }

  return 0;
}
