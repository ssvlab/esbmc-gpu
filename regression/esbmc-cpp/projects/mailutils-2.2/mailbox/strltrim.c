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
#include <string.h>
#include <mailutils/types.h>
#include <mailutils/cctype.h>
#include <mailutils/cstr.h>

size_t
mu_ltrim_class (char *str, int class)
{
  size_t i, len;
  
  if (!*str)
    return 0;
  len = strlen (str);
  
  for (i = 0; i < len && mu_c_is_class (str[i], class); i++)
    ;
  if (i)
    {
      len -= i;
      memmove (str, str + i, len + 1);
    }
  return len;
}

size_t
mu_ltrim_cset (char *str, const char *cset)
{
  size_t i, len;
  
  if (!*str)
    return 0;
  len = strlen (str);
  
  for (i = 0; i < len && strchr (cset, str[i]) != NULL; i++)
    ;
  if (i)
    {
      len -= i;
      memmove (str, str + i, len + 1);
    }
  return len;
}

