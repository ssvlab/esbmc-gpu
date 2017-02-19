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

#ifndef _MAILUTILS_KWD_H
#define _MAILUTILS_KWD_H

typedef struct mu_kwd mu_kwd_t;

struct mu_kwd
{
  char *name;
  int tok;
};

int mu_kwd_xlat_name (mu_kwd_t *kwtab, const char *str, int *pres);
int mu_kwd_xlat_name_ci (mu_kwd_t *kwtab, const char *str, int *pres);
int mu_kwd_xlat_name_len (mu_kwd_t *kwtab, const char *str, size_t len,
			  int *pres);
int mu_kwd_xlat_name_len_ci (mu_kwd_t *kwtab, const char *str, size_t len,
			     int *pres);
int mu_kwd_xlat_tok (mu_kwd_t *kwtab, int tok, const char **pres);

#endif
