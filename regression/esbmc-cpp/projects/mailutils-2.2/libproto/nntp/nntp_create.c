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

#include <stdlib.h>
#include <errno.h>
#include <mailutils/errno.h>
#include <mailutils/sys/nntp.h>

/* Initialise a mu_nntp_t handle.  */

int
mu_nntp_create (mu_nntp_t *pnntp)
{
  mu_nntp_t nntp;

  /* Sanity check.  */
  if (pnntp == NULL)
    return EINVAL;

  nntp = calloc (1, sizeof *nntp);
  if (nntp == NULL)
    return ENOMEM;

  /* Reserve space for the ack(nowledgement) response.
     According to RFC 977: The maximum length of the first line of a
     command response (including the initial greeting) is unchanged at
     512 octets (including the terminating CRLF).  */
  nntp->ack.len = 512;
  nntp->ack.buf = calloc (nntp->ack.len, 1);
  if (nntp->ack.buf == NULL)
    {
      mu_nntp_destroy (&nntp);
      return ENOMEM;
    }
  nntp->ack.ptr = nntp->ack.buf;

  /* Reserve space for the data response/content.
     RFC 977 recommands 255, but we grow it as needed.  */
  nntp->io.len = 255;
  nntp->io.buf = calloc (nntp->io.len, 1);
  if (nntp->io.buf == NULL)
    {
      mu_nntp_destroy (&nntp);
      return ENOMEM;
    }
  nntp->io.ptr = nntp->io.buf;

  nntp->state = MU_NNTP_NO_STATE; /* Init with no state.  */
  nntp->timeout = (10 * 60) * 100; /* The default Timeout is 10 minutes.  */
  nntp->acknowledge = 0; /* No Ack received.  */

  *pnntp = nntp;
  return 0; /* Okdoke.  */
}
