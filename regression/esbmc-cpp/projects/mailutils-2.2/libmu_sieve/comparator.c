/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2007, 2008, 2009,
   2010 Free Software Foundation, Inc.

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
#include <fnmatch.h>
#include <regex.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>

typedef struct {
  const char *name;
  int required;
  mu_sieve_comparator_t comp[MU_SIEVE_MATCH_LAST];
} sieve_comparator_record_t;

int
mu_sieve_register_comparator (mu_sieve_machine_t mach,
			   const char *name,
			   int required,
			   mu_sieve_comparator_t is,
			   mu_sieve_comparator_t contains,
			   mu_sieve_comparator_t matches,
			   mu_sieve_comparator_t regex,
			   mu_sieve_comparator_t eq)
{
  sieve_comparator_record_t *rp;

  if (!mach->comp_list)
    {
      int rc = mu_list_create (&mach->comp_list);
      if (rc)
	return rc;
    }

  rp = mu_sieve_malloc (mach, sizeof (*rp));
  rp->required = required;
  rp->name = name;
  rp->comp[MU_SIEVE_MATCH_IS] = is;       
  rp->comp[MU_SIEVE_MATCH_CONTAINS] = contains; 
  rp->comp[MU_SIEVE_MATCH_MATCHES] = matches;  
  rp->comp[MU_SIEVE_MATCH_REGEX] = regex;    
  rp->comp[MU_SIEVE_MATCH_EQ] = eq;    

  return mu_list_append (mach->comp_list, rp);
}

sieve_comparator_record_t *
_lookup (mu_list_t list, const char *name)
{
  mu_iterator_t itr;
  sieve_comparator_record_t *reg;

  if (!list || mu_list_get_iterator (list, &itr))
    return NULL;

  for (mu_iterator_first (itr); !mu_iterator_is_done (itr); mu_iterator_next (itr))
    {
      mu_iterator_current (itr, (void **)&reg);
      if (strcmp (reg->name, name) == 0)
	break;
      else
	reg = NULL;
    }
  mu_iterator_destroy (&itr);
  return reg;
}
    
int
mu_sieve_require_comparator (mu_sieve_machine_t mach, const char *name)
{
  sieve_comparator_record_t *reg = _lookup (mach->comp_list, name);
  if (!reg)
    {
      if (!(mu_sieve_load_ext (mach, name) == 0
	    && (reg = _lookup (mach->comp_list, name)) != NULL))
	return 1;
    }

  reg->required = 1;
  return 0;
}

mu_sieve_comparator_t 
mu_sieve_comparator_lookup (mu_sieve_machine_t mach, const char *name, 
                            int matchtype)
{
  sieve_comparator_record_t *reg = _lookup (mach->comp_list, name);
  if (reg && reg->comp[matchtype])
    return reg->comp[matchtype];
  return NULL;
}

static int
_find_comparator (void *item, void *data)
{
  mu_sieve_runtime_tag_t *tag = item;

  if (strcmp (tag->tag, TAG_COMPFUN) == 0)
    {
      *(mu_sieve_comparator_t*)data = tag->arg->v.ptr;
      return 1;
    }
  return 0;
}

mu_sieve_comparator_t
mu_sieve_get_comparator (mu_sieve_machine_t mach, mu_list_t tags)
{
  mu_sieve_comparator_t comp = NULL;

  mu_list_do (tags, _find_comparator, &comp);
  return comp ? comp : mu_sieve_comparator_lookup (mach,
						"i;ascii-casemap",
						MU_SIEVE_MATCH_IS);
}

/* Compile time support */

struct regex_data {
  int flags;
  mu_list_t list;
};

#ifndef FNM_CASEFOLD
static int
_pattern_upcase (void *item, void *data)
{
  mu_strupper (item);
  return 0;
}
#endif

static int
_regex_compile (void *item, void *data)
{
  struct regex_data *rd = data;
  int rc;
  regex_t *preg = mu_sieve_malloc (mu_sieve_machine, sizeof (*preg));
  
  rc = regcomp (preg, (char*)item, rd->flags);
  if (rc)
    {
      size_t size = regerror (rc, preg, NULL, 0);
      char *errbuf = malloc (size + 1);
      if (errbuf)
	{
	  regerror (rc, preg, errbuf, size);
	  mu_sv_compile_error (&mu_sieve_locus, _("regex error: %s"), errbuf);
	  free (errbuf);
	}
      else
	mu_sv_compile_error (&mu_sieve_locus, _("regex error"));
      return rc;
    }

  mu_list_append (rd->list, preg);
  
  return 0;
}

static int
_free_regex (void *item, void *unused)
{
  regfree ((regex_t*)item);
  return 0;
}

static void
_free_reglist (void *data)
{
  mu_list_t list = data;
  mu_list_do (list, _free_regex, NULL);
  mu_list_destroy (&list);
}

static int
comp_false (const char *pattern, const char *text)
{
  return 0;
}

int
mu_sieve_match_part_checker (const char *name, mu_list_t tags, mu_list_t args)
{
  mu_iterator_t itr;
  mu_sieve_runtime_tag_t *match = NULL;
  mu_sieve_runtime_tag_t *comp = NULL;
  mu_sieve_runtime_tag_t *tmp;
  mu_sieve_comparator_t compfun = NULL;
  char *compname = "false";
  
  int matchtype;
  int err = 0;
  
  if (!tags || mu_list_get_iterator (tags, &itr))
    return 0;

  for (mu_iterator_first (itr); !err && !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      mu_sieve_runtime_tag_t *t;
      mu_iterator_current (itr, (void **)&t);
      
      if (strcmp (t->tag, "is") == 0
	  || strcmp (t->tag, "contains") == 0
	  || strcmp (t->tag, "matches") == 0
	  || strcmp (t->tag, "regex") == 0
	  || strcmp (t->tag, "count") == 0
	  || strcmp (t->tag, "value") == 0)
	{
	  if (match)
	    {
	      mu_sv_compile_error (&mu_sieve_locus, 
			     _("match type specified twice in call to `%s'"),
				   name);
	      err = 1;
	    }
	  else
	    match = t;
	}
      else if (strcmp (t->tag, "comparator") == 0) 
	comp = t;
    }

  mu_iterator_destroy (&itr);

  if (err)
    return 1;

  if (!match || strcmp (match->tag, "is") == 0)
    matchtype = MU_SIEVE_MATCH_IS;
  else if (strcmp (match->tag, "contains") == 0)
    matchtype = MU_SIEVE_MATCH_CONTAINS;
  else if (strcmp (match->tag, "matches") == 0)
    matchtype = MU_SIEVE_MATCH_MATCHES;
  else if (strcmp (match->tag, "regex") == 0)
    matchtype = MU_SIEVE_MATCH_REGEX;
  else
    {
      char *str = match->arg->v.string;
      if (strcmp (match->tag, "count") == 0)
	{
	  mu_sieve_value_t *val;
	  char *str;
	  size_t count;
	  
	  if (comp && strcmp (comp->arg->v.string, "i;ascii-numeric"))
	    {
	      mu_sv_compile_error (&mu_sieve_locus, 
				   /* TRANSLATORS: Do not translate ':count'.
				      It is the name of a Sieve tag */
				   _("comparator %s is incompatible with "
				     ":count in call to `%s'"),
				   comp->arg->v.string,
				   name);
	      return 1;
	    }

          matchtype = MU_SIEVE_MATCH_LAST; /* to not leave it undefined */
	  compfun = comp_false;
	  val = mu_sieve_value_get (args, 1);
	  if (!val)
	    return 1; /* shouldn't happen */
	  /* NOTE: Type of v is always SVT_STRING_LIST */
	  mu_list_count (val->v.list, &count);
	  if (count > 1)
	    {
	      mu_sv_compile_error (&mu_sieve_locus, 
			_("second argument must be a list of one element"));
	      return 1;
	    }
	  mu_list_get (val->v.list, 0, (void **) &str);
	  count = strtoul (str, &str, 10);
	  if (*str)
	    {
	      mu_sv_compile_error (&mu_sieve_locus, 
			   _("second argument cannot be converted to number"));
	      return 1;
	    }
	}
      else
	matchtype = MU_SIEVE_MATCH_EQ;

      if (mu_sieve_str_to_relcmp (str, NULL, NULL))
	{
	  mu_sv_compile_error (&mu_sieve_locus, 
			       _("invalid relational match `%s' in call to `%s'"),
			       str, name);
	  return 1;
	}
    }

  if (!compfun)
    {
      compname = comp ? comp->arg->v.string : "i;ascii-casemap";
      compfun = mu_sieve_comparator_lookup (mu_sieve_machine, compname, 
                                            matchtype);
      if (!compfun)
	{
	  mu_sv_compile_error (&mu_sieve_locus, 
			   _("comparator `%s' is incompatible with match type `%s' in call to `%s'"),
			       compname, match ? match->tag : "is", name);
	  return 1;
	}
    }

  tmp = mu_sieve_malloc (mu_sieve_machine, sizeof (*tmp));
  tmp->tag = TAG_COMPFUN;
  tmp->arg = mu_sieve_value_create (SVT_POINTER, compfun);
  mu_list_append (tags, tmp);
  
  if (matchtype == MU_SIEVE_MATCH_REGEX)
    {
      /* To speed up things, compile all patterns at once.
	 Notice that it is supposed that patterns are in arg 2 */
      mu_sieve_value_t *val, *newval;
      struct regex_data rd;
      int rc;
      
      if (mu_list_get (args, 1, (void**)&val))
	return 0;

      rd.flags = REG_EXTENDED;
      if (strcmp (compname, "i;ascii-casemap") == 0)
	rd.flags |= REG_ICASE;

      mu_list_create (&rd.list);
      
      rc = mu_sieve_vlist_do (val, _regex_compile, &rd);

      mu_sieve_machine_add_destructor (mu_sieve_machine, _free_reglist, 
                                       rd.list);

      if (rc)
	return rc;
      newval = mu_sieve_value_create (SVT_STRING_LIST, rd.list);
      mu_list_replace (args, val, newval);
    }
#ifndef FNM_CASEFOLD
  else if (matchtype == MU_SIEVE_MATCH_MATCHES
	   && strcmp (compname, "i;ascii-casemap") == 0)
    {
      int rc;
      mu_sieve_value_t *val;

      if (mu_list_get (args, 1, (void**)&val))
	return 0;
      rc = mu_sieve_vlist_do (val, _pattern_upcase, NULL);
      if (rc)
	return rc;
    }
#endif
  return 0;
}

/* Particular comparators */

/* :comparator i;octet */

static int
i_octet_is (const char *pattern, const char *text)
{
  return strcmp (pattern, text) == 0;
}

static int
i_octet_contains (const char *pattern, const char *text)
{
  return strstr (text, pattern) != NULL;
}

static int 
i_octet_matches (const char *pattern, const char *text)
{
  return fnmatch (pattern, text, 0) == 0;
}

static int
i_octet_regex (const char *pattern, const char *text)
{
  return regexec ((regex_t *) pattern, text, 0, NULL, 0) == 0;
}

static int
i_octet_eq (const char *pattern, const char *text)
{
  return strcmp (text, pattern);
}

/* :comparator i;ascii-casemap */
static int
i_ascii_casemap_is (const char *pattern, const char *text)
{
  return mu_c_strcasecmp (pattern, text) == 0;
}

static int
i_ascii_casemap_contains (const char *pattern, const char *text)
{
  return mu_strcasestr (text, pattern) != NULL;
}

static int
i_ascii_casemap_matches (const char *pattern, const char *text)
{
#ifdef FNM_CASEFOLD
  return fnmatch (pattern, text, FNM_CASEFOLD) == 0;
#else
  int rc;
  char *p = strdup (text);
  _pattern_upcase (p, NULL);
  rc = fnmatch (pattern, p, 0) == 0;
  free (p);
  return rc;
#endif
}

static int
i_ascii_casemap_regex (const char *pattern, const char *text)
{
  return regexec ((regex_t *) pattern, text, 0, NULL, 0) == 0;
}

static int
i_ascii_casemap_eq (const char *pattern, const char *text)
{
  return mu_c_strcasecmp (text, pattern);
}

/* :comparator i;ascii-numeric */
static int
i_ascii_numeric_is (const char *pattern, const char *text)
{
  if (mu_isdigit (*pattern))
    {
      if (mu_isdigit (*text))
	return strtol (pattern, NULL, 10) == strtol (text, NULL, 10);
      else 
	return 0;
    }
  else if (mu_isdigit (*text))
    return 0;
  else
    return 1;
}

static int
i_ascii_numeric_eq (const char *pattern, const char *text)
{
  if (mu_isdigit (*pattern))
    {
      if (mu_isdigit (*text))
	{
	  size_t a = strtoul (pattern, NULL, 10);
	  size_t b = strtoul (text, NULL, 10);
	  if (b > a)
	    return 1;
	  else if (b < a)
	    return -1;
	  return 0;
	}
      else 
	return 1;
    }
  else
    return 1;
}

void
mu_sv_register_standard_comparators (mu_sieve_machine_t mach)
{
  mu_sieve_register_comparator (mach,
			     "i;octet",
			     1,
			     i_octet_is,
			     i_octet_contains,
			     i_octet_matches,
			     i_octet_regex,
			     i_octet_eq);
  mu_sieve_register_comparator (mach,
			     "i;ascii-casemap",
			     1,
			     i_ascii_casemap_is,
			     i_ascii_casemap_contains,
			     i_ascii_casemap_matches,
			     i_ascii_casemap_regex,
			     i_ascii_casemap_eq);
  mu_sieve_register_comparator (mach,
			     "i;ascii-numeric",
			     0,
			     i_ascii_numeric_is,
			     NULL,
			     NULL,
			     NULL,
			     i_ascii_numeric_eq);
}
