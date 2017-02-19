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

#include <stdlib.h>
#include <errno.h>
#include <mailutils/errno.h>
#include <mailutils/sys/pop3.h>

/* Initialise a mu_pop3_t handle.  */

int
mu_pop3_create (mu_pop3_t *ppop3)
{
  mu_pop3_t pop3;

  /* Sanity check.  */
  if (ppop3 == NULL)
    return EINVAL;

  pop3 = calloc (1, sizeof *pop3);
  if (pop3 == NULL)
    return ENOMEM;

  /* Reserve space for the ack(nowledgement) response.
     According to RFC 2449: The maximum length of the first line of a
     command response (including the initial greeting) is unchanged at
     512 octets (including the terminating CRLF).  */
  pop3->ack.len = 512;
  pop3->ack.buf = calloc (pop3->ack.len, 1);
  if (pop3->ack.buf == NULL)
    {
      mu_pop3_destroy (&pop3);
      return ENOMEM;
    }
  pop3->ack.ptr = pop3->ack.buf;

  /* Reserve space for the data response/content.
     RFC 2449 recommands 255, but we grow it as needed.  */
  pop3->io.len = 255;
  pop3->io.buf = calloc (pop3->io.len, 1);
  if (pop3->io.buf == NULL)
    {
      mu_pop3_destroy (&pop3);
      return ENOMEM;
    }
  pop3->io.ptr = pop3->io.buf;

  pop3->state = MU_POP3_NO_STATE; /* Init with no state.  */
  pop3->timeout = (10 * 60) * 100; /* The default Timeout is 10 minutes.  */
  pop3->acknowledge = 0; /* No Ack received.  */

  *ppop3 = pop3;
  return 0; /* Okdoke.  */
}
