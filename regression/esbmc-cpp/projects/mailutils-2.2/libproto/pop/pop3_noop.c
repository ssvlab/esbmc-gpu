/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software Foundation,
   Inc.

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

int
mu_pop3_noop (mu_pop3_t pop3)
{
  int status;

  if (pop3 == NULL)
    return EINVAL;

  switch (pop3->state)
    {
    case MU_POP3_NO_STATE:
      status = mu_pop3_writeline (pop3, "NOOP\r\n");
      MU_POP3_CHECK_ERROR (pop3, status);
      mu_pop3_debug_cmd (pop3);
      pop3->state = MU_POP3_NOOP;

    case MU_POP3_NOOP:
      status = mu_pop3_send (pop3);
      MU_POP3_CHECK_EAGAIN (pop3, status);
      pop3->acknowledge = 0;
      pop3->state = MU_POP3_NOOP_ACK;

    case MU_POP3_NOOP_ACK:
      status = mu_pop3_response (pop3, NULL, 0, NULL);
      MU_POP3_CHECK_EAGAIN (pop3, status);
      mu_pop3_debug_ack (pop3);
      MU_POP3_CHECK_OK (pop3);
      pop3->state = MU_POP3_NO_STATE;
      break;

      /* They must deal with the error first by reopening.  */
    case MU_POP3_ERROR:
      status = ECANCELED;
      break;

    default:
      status = EINPROGRESS;
    }

  return status;
}
