/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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
#include <mailutils/stream.h>
#include <mailutils/mutil.h>
#include <mailutils/errno.h>

#define FILE_PERM_READ 0x1
#define FILE_PERM_WRITE 0x2

static int
parse_perm_bits (int *pmode, const char *str, const char **endp)
{
  switch (*str)
    {
    case '+':
    case '=':
      str++;
      break;

    default:
      if (endp)
	*endp = str;
      return 1;
    }

  for (; *str; str++)
    {
      switch (*str)
	{
	case 'r':
	  *pmode |= FILE_PERM_READ;
	  break;

	case 'w':
	  *pmode |= FILE_PERM_WRITE;
	  break;

	case ',':
	  if (endp)
	    *endp = str;
	  return 0;
	  
	default:
	  if (endp)
	    *endp = str;
	  return 1;
	}
    }
  if (endp)
    *endp = str;
  return 0;
}

/* Parse a MU stream permission specification in form:

      g(+|=)[wr]+,o(+|=)[wr]+

   Return 0 on success.
   On failure, return MU_ERR_FAILURE and point endp to the offending character.
*/
int
mu_parse_stream_perm_string (int *pmode, const char *str, const char **endp)
{
  int mode = 0;
  int f;
  while (*str)
    {
      switch (*str)
	{
	case 'g':
	  if (parse_perm_bits (&f, str + 1, &str))
	    {
	      if (endp)
		*endp = str;
	      return MU_ERR_FAILURE;
	    }
	  if (f & FILE_PERM_READ)
	    mode |= MU_STREAM_IRGRP;
	  if (f & FILE_PERM_WRITE)
	    mode |= MU_STREAM_IWGRP;
	  break;
	  
	case 'o':
	  if (parse_perm_bits (&f, str + 1, &str))
	    {
	      if (endp)
		*endp = str;
	      return MU_ERR_FAILURE;
	    }
	  if (f & FILE_PERM_READ)
	    mode |= MU_STREAM_IROTH;
	  if (f & FILE_PERM_WRITE)
	    mode |= MU_STREAM_IWOTH;
	  break;
	  
	default:
	  if (endp)
	    *endp = str;
	  return MU_ERR_FAILURE;
	}
      if (*str == ',')
	str++;
    }
  *pmode = mode;
  if (endp)
    *endp = str;
  return 0;
}
      
