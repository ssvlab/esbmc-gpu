/* This file is part of GNU Mailutils
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>

/*
 * removes whitespace from the beginning and end of a string
 */
char *
mu_str_stripws (char *string)
{
  mu_rtrim_class (string, MU_CTYPE_SPACE);
  return mu_str_skip_class (string, MU_CTYPE_SPACE);
}
