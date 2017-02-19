/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2010 Free Software Foundation, Inc.

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

#include <string.h>
#include <errno.h>
#include <mailutils/sys/pop3.h>

/* If we did not grap the ack already, call pop3_readline() but handle
   Nonblocking also.  */
int
mu_pop3_response (mu_pop3_t pop3, char *buffer, size_t buflen, size_t *pnread)
{
  size_t n = 0;
  int status = 0;

  if (pop3 == NULL)
    return EINVAL;

  if (!pop3->acknowledge)
    {
      size_t len = pop3->ack.len - (pop3->ack.ptr  - pop3->ack.buf);
      status = mu_pop3_readline (pop3, pop3->ack.ptr, len, &n);
      pop3->ack.ptr += n;
      if (status == 0)
	{
	  len = pop3->ack.ptr - pop3->ack.buf;
	  if (len && pop3->ack.buf[len - 1] == '\n')
	    pop3->ack.buf[len - 1] = '\0';
	  pop3->acknowledge = 1; /* Flag that we have the ack.  */
	  pop3->ack.ptr = pop3->ack.buf;
	}
      else
	{
	  /* Provide them with an error.  */
	  const char *econ = "-ERR POP3 IO ERROR";
	  n = strlen (econ);
	  strcpy (pop3->ack.buf, econ);
	}
    }
  else
    n = strlen (pop3->ack.buf);

  if (buffer)
    {
      buflen--; /* Leave space for the NULL.  */
      n = (buflen < n) ? buflen : n;
      memcpy (buffer, pop3->ack.buf, n);
      buffer[n] = '\0';
    }

  if (pnread)
    *pnread = n;
  return status;
}
