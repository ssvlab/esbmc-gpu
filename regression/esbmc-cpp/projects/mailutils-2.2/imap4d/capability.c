/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2003, 2005, 2007, 2008, 2010 Free Software
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

#include "imap4d.h"

static mu_list_t capa_list;

static int
comp (const void *item, const void *data)
{
  return strcmp ((char*)item, (char*)data);
}

void
imap4d_capability_add (const char *str)
{
  if (!capa_list)
    {
      mu_list_create (&capa_list);
      mu_list_set_comparator (capa_list, comp);
    }
  mu_list_append (capa_list, (void*)str);
}

void
imap4d_capability_remove (const char *str)
{
  mu_list_remove (capa_list, (void*)str);
}

void
imap4d_capability_init ()
{
  static char *capa[] = {
    "IMAP4rev1",
    "NAMESPACE",
    "ID",
    "IDLE",
    "LITERAL+",
    "UNSELECT",
    NULL
  };
  int i;

  for (i = 0; capa[i]; i++)
    imap4d_capability_add (capa[i]);
}

static int
print_capa (void *item, void *data)
{
  util_send (" %s", (char *)item);
  return 0;
}

int
imap4d_capability (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  if (imap4d_tokbuf_argc (tok) != 2)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  util_send ("* CAPABILITY");

  mu_list_do (capa_list, print_capa, NULL);
  
  imap4d_auth_capability ();
  util_send ("\r\n");

  return util_finish (command, RESP_OK, "Completed");
}
