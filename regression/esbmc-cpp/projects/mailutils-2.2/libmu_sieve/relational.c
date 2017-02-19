/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2008, 2010 Free Software Foundation,
   Inc.

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

#include <unistd.h>  
#include <string.h>  
#include <sieve-priv.h>

#define CAT2(a,b) a##b
#define CAT3(a,b,c) a##b##c

#define DCL(name,op)\
 static int CAT2(op_,name) (int num, int lim) { return num op lim; }\
 static int CAT3(op_,name,_size_t) (size_t num, size_t lim) { return num op lim; }

DCL(eq,==)
DCL(ne,!=)
DCL(gt,>)
DCL(ge,>=)
DCL(lt,<)
DCL(le,<=)

static struct reltest_tab {
  char *name;
  mu_sieve_relcmp_t test;
  mu_sieve_relcmpn_t stest;
} testtab[] = {
#define DEF(name) { #name, CAT2(op_,name), CAT3(op_,name,_size_t) }

  DEF(eq),
  DEF(ne),
  DEF(gt),
  DEF(ge),
  DEF(lt),
  DEF(le)
};

static struct reltest_tab *
_relcmp_lookup (const char *str)
{
  int i;

  for (i = 0; i < sizeof(testtab) / sizeof(testtab[0]); i++)
    if (strcmp (testtab[i].name, str) == 0)
      return &testtab[i];
  return 0;
}

int
mu_sieve_str_to_relcmp (const char *str,
		      mu_sieve_relcmp_t *test, mu_sieve_relcmpn_t *stest)
{
  struct reltest_tab *t = _relcmp_lookup (str);
  if (t)
    {
      if (test)
	*test = t->test;
      if (stest)
	*stest = t->stest;
      return 0;
    }
  return 1;
}

mu_sieve_relcmp_t
mu_sieve_get_relcmp (mu_sieve_machine_t mach, mu_list_t tags)
{
  mu_sieve_value_t *arg;
  mu_sieve_relcmp_t test = NULL;
  
  if (mu_sieve_tag_lookup (tags, "value", &arg) == 0)
    return op_ne;
  mu_sieve_str_to_relcmp (arg->v.string, &test, NULL);
  return test;
}

int
mu_sieve_require_relational (mu_sieve_machine_t mach, const char *name)
{
  /* noop */
  return 0;
}
