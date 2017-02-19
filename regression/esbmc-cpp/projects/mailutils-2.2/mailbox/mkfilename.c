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
   Public License along with this library.  If not,
   see <http://www.gnu.org/licenses/>. */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/alloc.h>
#include <mailutils/mutil.h>

char *
mu_make_file_name (const char *dir, const char *file)
{
  char *tmp;
  size_t dirlen = strlen (dir);
  size_t len;

  while (dirlen > 0 && dir[dirlen-1] == '/')
    dirlen--;
  
  len = dirlen + 1 + strlen (file);
  tmp = mu_alloc (len + 1);
  if (tmp)
    {
      memcpy (tmp, dir, dirlen);
      tmp[dirlen++] = '/';
      strcpy (tmp + dirlen, file);
    }
  return tmp;
}
