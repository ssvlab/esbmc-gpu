/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <mailutils/error.h>
#include <mailutils/sys/nntp.h>

/*
  LIST EXTENSIONS command, return a iterator that contains the result.
  It is the responsability of the caller to destroy the iterator(mu_iterator_destroy).
 */
int
mu_nntp_list_active (mu_nntp_t nntp, const char *wildmat, mu_iterator_t *piterator)
{
  int status = 0;

  if (nntp == NULL)
    return EINVAL;
  if (piterator == NULL)
    return MU_ERR_OUT_PTR_NULL;

  switch (nntp->state)
    {
    case MU_NNTP_NO_STATE:
      if (wildmat == NULL || *wildmat == '\0')
	status = mu_nntp_writeline (nntp, "LIST ACTIVE\r\n");
      else
	status = mu_nntp_writeline (nntp, "LIST ACTIVE %s\r\n", wildmat);
      MU_NNTP_CHECK_ERROR (nntp, status);
      mu_nntp_debug_cmd (nntp);
      nntp->state = MU_NNTP_LIST_ACTIVE;

    case MU_NNTP_LIST_ACTIVE:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->acknowledge = 0;
      nntp->state = MU_NNTP_LIST_ACTIVE_ACK;

    case MU_NNTP_LIST_ACTIVE_ACK:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE (nntp, MU_NNTP_RESP_CODE_LIST_FOLLOW);
      status = mu_nntp_iterator_create (nntp, piterator);
      MU_NNTP_CHECK_ERROR(nntp, status);
      nntp->state = MU_NNTP_LIST_ACTIVE_RX;

    case MU_NNTP_LIST_ACTIVE_RX:
      break;

      /* They must deal with the error first by reopening.  */
    case MU_NNTP_ERROR:
      status = ECANCELED;
      break;

    default:
      status = EINPROGRESS;
    }

  return status;
}

int
mu_nntp_parse_list_active (const char *buffer, char **group, unsigned long *high, unsigned long *low, char *status)
{
  unsigned long dummy;
  char cdummy;
  char *name;

  if (buffer == NULL || *buffer == '\0')
    return EINVAL;

  name = calloc(512, 1);
  if (name == NULL)
    return ENOMEM;

  if (high == NULL)
    high = &dummy;
  if (low == NULL)
    low = &dummy;
  if (status == NULL)
    status = &cdummy;

  sscanf (buffer, "%511s %ld %ld %c", name, high, low, status);

  if (group == NULL)
    free (name);
  else
    *group = name;
  return 0;
}
