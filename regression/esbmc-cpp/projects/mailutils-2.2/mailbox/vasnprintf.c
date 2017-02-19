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
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <mailutils/io.h>

int
mu_vasnprintf (char **pbuf, size_t *psize, const char *fmt, va_list ap)
{
  char *buf = *pbuf;
  size_t buflen = *psize;
  int rc = 0;
  
  if (!buf)
    {
      if (buflen == 0)
	buflen = 512; /* Initial allocation */
      
      buf = calloc (1, buflen);
      if (buf == NULL)
	return ENOMEM;
    }
  
  for (;;)
    {
      ssize_t n = vsnprintf (buf, buflen, fmt, ap);
      if (n < 0 || n >= buflen || !memchr (buf, '\0', n + 1))
	{
	  char *newbuf;
	  size_t newlen = buflen * 2;
	  if (newlen < buflen)
	    {
	      rc = ENOMEM;
	      break;
	    }
	  newbuf = realloc (buf, newlen);
	  if (newbuf == NULL)
	    {
	      rc = ENOMEM;
	      break;
	    }
	  buflen = newlen;
	  buf = newbuf;
	}
      else
	break;
    }

  if (rc)
    {
      if (!*pbuf)
	{
	  /* We made first allocation, now free it */
	  free (buf);
	  buf = NULL;
	  buflen = 0;
	}
    }

  *pbuf = buf;
  *psize = buflen;
  return rc;
}
