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
#include <string.h>
#include <errno.h>
#include <mailutils/sys/nntp.h>

int
mu_nntp_disconnect (mu_nntp_t nntp)
{
  /* Sanity checks.  */
  if (nntp == NULL)
    return EINVAL;

  /* We can keep some of the fields, if they decide to nntp_connect() again but
     clear the states.  */
  nntp->state = MU_NNTP_NO_STATE;
  nntp->acknowledge = 0;

  /* Clear the buffers.  */
  memset (nntp->io.buf, '\0', nntp->io.len);
  nntp->io.ptr = nntp->io.buf;
  memset (nntp->ack.buf, '\0', nntp->ack.len);
  nntp->ack.ptr = nntp->ack.buf;

  /* Close the stream.  */
  return mu_stream_close (nntp->carrier);
}
