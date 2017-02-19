/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2006, 2007, 2010 Free Software
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

typedef struct _alias *alias_t;

struct _alias
{
  mu_list_t list;
};

static mu_assoc_t aliases;

static void
alias_free (void *data)
{
  alias_t al = data;
  util_slist_destroy (&al->list);
}

static void
alias_print_group (const char *name, alias_t al)
{
  fprintf (ofile, "%s    ", name);
  util_slist_print (al->list, 0);
  fprintf (ofile, "\n");
}

static alias_t
alias_lookup (const char *name)
{
  return mu_assoc_ref (aliases, name);
}

static void
alias_print (char *name)
{
  if (!name)
    {
      mu_iterator_t itr;

      if (!aliases)
	return;

      mu_assoc_get_iterator (aliases, &itr);
      for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
	   mu_iterator_next (itr))
	{
	  const char *name;
	  alias_t al;
	  if (mu_iterator_current_kv (itr, (const void **)&name, (void**)&al))
	    continue;
	  alias_print_group (name, al);
	}
    }
  else
    {
      alias_t al;

      al = alias_lookup (name);
      if (!al)
	{
	  util_error (_("\"%s\": not a group"), name);
	  return;
	}
      alias_print_group (name, al);
    }
}

static int
alias_create (const char *name, alias_t *al)
{
  int rc;

  if (!aliases)
    {
      mu_assoc_create (&aliases, sizeof (struct _alias), 0);
      mu_assoc_set_free (aliases, alias_free);
    }
  
  rc = mu_assoc_ref_install (aliases, name, (void**) al);
  if (rc == MU_ERR_EXISTS)
    return 0;
  if (rc == 0)
    return mu_list_create (&(*al)->list);
  return 1;
}

void
alias_destroy (const char *name)
{
  mu_assoc_remove (aliases, name);
}


static void
recursive_alias_expand (const char *name, mu_list_t exlist, mu_list_t origlist)
{ 
  alias_t al;
  mu_iterator_t itr;
  
  if ((al = alias_lookup (name)) == NULL)
    {
      if (mu_list_locate (exlist, (void*)name, NULL) == MU_ERR_NOENT)
	mu_list_append (exlist, (void*)name);
      return;
    }
  
  mu_list_get_iterator (al->list, &itr);
  for (mu_iterator_first (itr);
       !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *word;
      
      mu_iterator_current (itr, (void **)&word);
      if (mu_list_locate (origlist, word, NULL) == MU_ERR_NOENT)
	{
	  mu_list_prepend (origlist, word);
	  recursive_alias_expand (word, exlist, origlist);
	  mu_list_remove (origlist, word);
	}
    }
  mu_iterator_destroy (&itr);
}

static int
string_comp (const void *item, const void *value)
{
  return strcmp (item, value);
}

char *
alias_expand (const char *name)
{
  alias_t al;
  mu_list_t list;
  
  if (mailvar_get (NULL, "recursivealiases", mailvar_type_boolean, 0) == 0)
    {
      char *s;
      mu_list_t origlist;
      
      int status = mu_list_create (&list);
      if (status)
	{
	  mu_diag_funcall (MU_DIAG_ERROR, "mu_list_create", NULL, status);
	  return NULL;
	}
      status = mu_list_create (&origlist);
      if (status)
	{
	  mu_list_destroy (&origlist);
	  mu_diag_funcall (MU_DIAG_ERROR, "mu_list_create", NULL, status);
	  return NULL;
	}
      mu_list_set_comparator (list, string_comp);
      mu_list_set_comparator (origlist, string_comp);
      recursive_alias_expand (name, list, origlist);
      s = util_slist_to_string (list, ",");
      mu_list_destroy (&origlist);
      mu_list_destroy (&list);
      return s;
    }
  
  if ((al = alias_lookup (name)) == NULL)
    return NULL;
  return util_slist_to_string (al->list, ",");
}


struct alias_iterator
{
  mu_iterator_t itr;
  const char *prefix;
  int prefixlen;
  int pos;
};

const char *
alias_iterate_next (alias_iterator_t atr)
{
  while (!mu_iterator_is_done (atr->itr))
    {
      const char *name;
      alias_t al;

      if (mu_iterator_current_kv (atr->itr, (const void **)&name, (void**)&al))
	continue;
      mu_iterator_next (atr->itr);
      if (strlen (name) >= atr->prefixlen
	  && strncmp (name, atr->prefix, atr->prefixlen) == 0)
	return name;
    }
  return NULL;
}

const char *
alias_iterate_first (const char *prefix, alias_iterator_t *pc)
{
  mu_iterator_t itr;
  alias_iterator_t atr;
  
  if (!aliases)
    {
      *pc = NULL;
      return NULL;
    }

  if (mu_assoc_get_iterator (aliases, &itr))
    return NULL;
  mu_iterator_first (itr);
  atr = xmalloc (sizeof *atr);
  atr->prefix = prefix;
  atr->prefixlen = strlen (prefix);
  atr->pos = 0;
  atr->itr = itr;
  *pc = atr;

  return alias_iterate_next (atr);
}

void
alias_iterate_end (alias_iterator_t *pc)
{
  mu_iterator_destroy (&(*pc)->itr);
  free (*pc);
  *pc = NULL;
}



/*
 * a[lias] [alias [address...]]
 * g[roup] [alias [address...]]
 */

int
mail_alias (int argc, char **argv)
{
  if (argc == 1)
    alias_print (NULL);
  else if (argc == 2)
    alias_print (argv[1]);
  else
    {
      alias_t al;

      if (alias_create (argv[1], &al))
	return 1;

      argc--;
      argv++;
      while (--argc)
	util_slist_add (&al->list, *++argv);
    }
  return 0;
}


