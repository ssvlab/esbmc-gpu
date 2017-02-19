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
  LIST xxxx command, return an iterator that contains the result.
  It is the responsability of the caller to destroy the iterator(mu_iterator_destroy).
 */
int
mu_nntp_list_distrib_pats (mu_nntp_t nntp, mu_iterator_t *piterator)
{
  int status = 0;

  if (nntp == NULL)
    return EINVAL;
  if (piterator == NULL)
    return MU_ERR_OUT_PTR_NULL;

  switch (nntp->state)
    {
    case MU_NNTP_NO_STATE:
      status = mu_nntp_writeline (nntp, "LIST DISTRIB.PATS\r\n");
      MU_NNTP_CHECK_ERROR (nntp, status);
      mu_nntp_debug_cmd (nntp);
      nntp->state = MU_NNTP_LIST_DISTRIB_PATS;

    case MU_NNTP_LIST_DISTRIB_PATS:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->acknowledge = 0;
      nntp->state = MU_NNTP_LIST_DISTRIB_PATS_ACK;

    case MU_NNTP_LIST_DISTRIB_PATS_ACK:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE (nntp, MU_NNTP_RESP_CODE_LIST_FOLLOW);
      status = mu_nntp_iterator_create (nntp, piterator);
      MU_NNTP_CHECK_ERROR(nntp, status);
      nntp->state = MU_NNTP_LIST_DISTRIB_PATS_RX;

    case MU_NNTP_LIST_DISTRIB_PATS_RX:
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
mu_nntp_parse_list_distrib_pats (const char *buffer, unsigned long *weight, char **wildmat, char **distrib)
{
  char *w;
  char *d;
  unsigned long dummy;

  if (buffer == NULL || *buffer == '\0')
    return EINVAL;

  w = calloc(512, 1);
  if (w == NULL)
    return ENOMEM;

  d = calloc(512, 1);
  if (d == NULL)
    {
      free (w);
      return ENOMEM;
    }

  if (weight == NULL)
    weight = &dummy;

  sscanf (buffer, "%ld:%511s:%511s", weight, w, d);

  if (wildmat == NULL)
    free (w);
  else
    *wildmat = w;

  if (distrib == NULL)
    free (d);
  else
    *distrib = d;
  return 0;
}
