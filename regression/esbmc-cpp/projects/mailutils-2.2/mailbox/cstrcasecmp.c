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
#include <stddef.h>
#include <mailutils/cctype.h>

int
mu_c_strcasecmp (const char *a, const char *b)
{
  int d = 0;
  for (; d == 0; a++, b++)
    {
      int ac = (int) *a;
      int bc = (int) *b;
      if (ac == 0 || bc == 0) 
	return ac - bc;
      if (mu_isascii (ac) && mu_isascii (bc))
	d = mu_toupper (ac) - mu_toupper (bc);
      else
	d = ac - bc;
    }
  return d;
}
			
int
mu_c_strncasecmp (const char *a, const char *b, size_t n)
{
  int d = 0;
  for (; d == 0 && n > 0; a++, b++, n--)
    {
      int ac = (int) *a;
      int bc = (int) *b;
      if (ac == 0 || bc == 0) 
	return ac - bc;
      if (mu_isascii (ac) && mu_isascii (bc))
	d = mu_toupper (ac) - mu_toupper (bc);
      else
	d = ac - bc;
    }
  return d;
}
