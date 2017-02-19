/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2004, 2005, 2007, 2010 Free Software
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

static mu_list_t retained_headers = NULL;
static mu_list_t ignored_headers = NULL;
static mu_list_t unfolded_headers = NULL;
static mu_list_t sender_headers = NULL;

static int
process_list (int argc, char **argv,
	      mu_list_t *list,
	      void (*fun) (mu_list_t *, char *),
	      char *msg)
{
  if (argc == 1)
    {
      if (mu_list_is_empty (*list))
	fprintf (ofile, _(msg));
      else
	util_slist_print (*list, 1);
      return 0;
    }

  while (--argc)
    fun (list, *++argv);
  return 0;
}

/*
 * ret[ain] [heder-field...]
 */

int
mail_retain (int argc, char **argv)
{
  return process_list (argc, argv, &retained_headers,
		       util_slist_add,
		       N_("No fields are currently being retained\n"));
}

/*
 * di[scard] [header-field...]
 * ig[nore] [header-field...]
 */

int
mail_discard (int argc, char **argv)
{
  return process_list (argc, argv, &ignored_headers,
		       util_slist_add,
		       N_("No fields are currently being ignored\n"));
  return 0;
}

/*
 * unfold [header-field...]
 */

int
mail_unfold (int argc, char **argv)
{
  return process_list (argc, argv, &unfolded_headers,
		       util_slist_add,
		       N_("No fields are currently being unfolded\n"));
}

/*
 * nounfold [header-field...]
 */

int
mail_nounfold (int argc, char **argv)
{
  return process_list (argc, argv, &unfolded_headers,
		       util_slist_remove,
		       N_("No fields are currently being unfolded\n"));
}

int
mail_header_is_unfoldable (const char *str)
{
  return util_slist_lookup (unfolded_headers, str);
}

int
mail_header_is_visible (const char *str)
{
  if (retained_headers)
    return util_slist_lookup (retained_headers, str);
  else
    return !util_slist_lookup (ignored_headers, str);
}

/*
 * sender [header-field...]
 */

int
mail_sender (int argc, char **argv)
{
  return process_list (argc, argv, &sender_headers,
		       util_slist_add,
		       N_("Sender address is obtained from the envelope\n"));
}

int
mail_nosender (int argc, char **argv)
{
  if (argc == 1)
    {
      util_slist_destroy (&sender_headers);
      fprintf (ofile, _("Sender address is obtained from the envelope\n"));
    }
  else 
    while (--argc)
      util_slist_remove (&sender_headers, *++argv);
  return 0;
}


mu_address_t
get_sender_address (mu_message_t msg)
{
  mu_iterator_t itr;
  mu_header_t header = NULL;
  mu_address_t addr = NULL;

  if (mu_message_get_header (msg, &header))
    return NULL;
  
  if (!sender_headers || mu_list_get_iterator (sender_headers, &itr))
    return NULL;

  for (mu_iterator_first (itr); !addr && !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *name;
      char *buf = NULL;
      
      mu_iterator_current (itr, (void **)&name);
      if (mu_header_aget_value (header, name, &buf) == 0)
	mu_address_create (&addr, buf);
      free (buf);
    }
  mu_iterator_destroy (&itr);
  return addr;
}
  
