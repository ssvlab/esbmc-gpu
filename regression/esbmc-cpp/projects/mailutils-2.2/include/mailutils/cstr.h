/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library.  If not, see
   <http://www.gnu.org/licenses/>. */

#ifndef _MAILUTILS_CSTR_H
#define _MAILUTILS_CSTR_H

#ifdef __cplusplus
extern "C" {
#endif

int mu_strlower (char *);
int mu_strupper (char *);

int mu_c_strcasecmp (const char *a, const char *b);
int mu_c_strncasecmp (const char *a, const char *b, size_t n);

size_t mu_rtrim_class (char *str, int __class);
size_t mu_rtrim_cset (char *str, const char *cset);
size_t mu_ltrim_class (char *str, int __class);
size_t mu_ltrim_cset (char *str, const char *cset);

char *mu_str_skip_class (const char *str, int __class);
char *mu_str_skip_cset (const char *str, const char *cset);

char *mu_str_skip_class_comp (const char *str, int __class);
char *mu_str_skip_cset_comp (const char *str, const char *cset);

char *mu_str_stripws (char *string);  
  
#ifdef __cplusplus
}
#endif
  
#endif

