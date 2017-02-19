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
#include <mailutils/errno.h>
#include <mailutils/sys/pop3.h>

void
mu_pop3_destroy (mu_pop3_t *ppop3)
{
  if (ppop3 && *ppop3)
    {
      mu_pop3_t pop3 = *ppop3;

      /* Free the response buffer.  */
      if (pop3->ack.buf)
	free (pop3->ack.buf);

      /* Free the io buffer.  */
      if (pop3->io.buf)
	free (pop3->io.buf);

      /* Free the timestamp use for APOP.  */
      if (pop3->timestamp)
	free (pop3->timestamp);

      /* Release the carrier.  */
      if (pop3->carrier)
	mu_stream_destroy (&pop3->carrier, pop3);

      free (pop3);

      *ppop3 = NULL;
    }
}
