/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

#ifndef _MAILUTILS_VARTAB_H
#define _MAILUTILS_VARTAB_H

#include <mailutils/types.h>

typedef int (*mu_var_expansion_fp) (const char *name, void *data, char **p);
typedef void (*mu_var_free_fp) (void *data, char *value);

int mu_vartab_create (mu_vartab_t *pvar);
int mu_vartab_destroy (mu_vartab_t *pvar);
int mu_vartab_define (mu_vartab_t var, const char *name, const char *value,
		      int isstatic);
int mu_vartab_define_exp (mu_vartab_t var, const char *name,
			  mu_var_expansion_fp fun, mu_var_free_fp free,
			  void *data);
int mu_vartab_count (mu_vartab_t vt, size_t *pcount);
int mu_vartab_getvar (mu_vartab_t vt, const char *name, const char **pvalue);
int mu_vartab_expand (mu_vartab_t vt, const char *str, char **pres);
#endif
