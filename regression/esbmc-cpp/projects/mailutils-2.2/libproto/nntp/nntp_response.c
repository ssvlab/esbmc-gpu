/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2007, 2010 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include <errno.h>
#include <mailutils/sys/nntp.h>

/* If we did not grap the ack already, call nntp_readline() but handle
   Nonblocking also.  */
int
mu_nntp_response (mu_nntp_t nntp, char *buffer, size_t buflen, size_t *pnread)
{
  size_t n = 0;
  int status = 0;

  if (nntp == NULL)
    return EINVAL;

  if (!nntp->acknowledge)
    {
      size_t len = nntp->ack.len - (nntp->ack.ptr  - nntp->ack.buf);
      status = mu_nntp_readline (nntp, nntp->ack.ptr, len, &n);
      nntp->ack.ptr += n;
      if (status == 0)
	{
	  len = nntp->ack.ptr - nntp->ack.buf;
	  if (len && nntp->ack.buf[len - 1] == '\n')
	    nntp->ack.buf[len - 1] = '\0';
	  nntp->acknowledge = 1; /* Flag that we have the ack.  */
	  nntp->ack.ptr = nntp->ack.buf;
	}
      else
	{
	  /* Provide them with an error.  */
	  const char *econ = "500 NNTP IO ERROR";
	  n = strlen (econ);
	  strcpy (nntp->ack.buf, econ);
	}
    }
  else
    n = strlen (nntp->ack.buf);

  if (buffer)
    {
      buflen--; /* Leave space for the NULL.  */
      n = (buflen < n) ? buflen : n;
      memcpy (buffer, nntp->ack.buf, n);
      buffer[n] = '\0';
    }

  if (pnread)
    *pnread = n;
  return status;
}

int
mu_nntp_response_code(mu_nntp_t nntp)
{
  char buffer[4];
  int code;

  memset (buffer, '\0', 4);
  mu_nntp_response(nntp, buffer, 4, NULL);
  /* translate the number, basically strtol() without the overhead. */
  code = (buffer[0] - '0')*100 + (buffer[1] - '0')*10 + (buffer[2] - '0');
  return code;
}
