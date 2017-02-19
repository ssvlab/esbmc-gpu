/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2006, 2007, 2010 Free
   Software Foundation, Inc.

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

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <stdlib.h>

#include <mailutils/error.h>
#include <mailutils/errno.h>

#include <regex.h>

static regex_t *re_prefix;

int
mu_unre_set_regex (const char *str, int caseflag, char **errp)
{
  int rc;
  int flags = REG_EXTENDED;

  if (errp)
    *errp = NULL;

  if (!str)
    str = "^re: *";
  if (re_prefix)
    regfree (re_prefix);
  else
    {
      re_prefix = malloc (sizeof (*re_prefix));
      if (!re_prefix)
	return ENOMEM;
    }
  if (!caseflag)
    flags |= REG_ICASE;
  rc = regcomp (re_prefix, str, flags);
  if (rc)
    {
      if (errp)
	{
	  size_t s = regerror (rc, re_prefix, NULL, 0);
	  s++;
	  *errp = malloc (s);
	  if (*errp)
	    regerror (rc, re_prefix, *errp, s);
	}
      regfree (re_prefix);
      free (re_prefix);
      return MU_ERR_FAILURE;
    }
  return 0;
}

int
mu_unre_subject (const char *subject, const char **new_subject)
{
  int rc;
  regmatch_t rm;

  if (!subject)
    return EINVAL;
  
  if (!re_prefix)
    {
      rc = mu_unre_set_regex (NULL, 0, NULL);
      if (rc)
	return rc;
    }

  rc = regexec (re_prefix, subject, 1, &rm, 0);
  if (rc == 0 && rm.rm_eo != -1 && new_subject)
    *new_subject = subject + rm.rm_eo;
  return rc;
}
