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
#include <errno.h>
#include <stdlib.h>
#include <mailutils/sys/nntp.h>

static int mu_nntp_parse_group(mu_nntp_t nntp, int code, unsigned long *ptotal, unsigned long *plow, unsigned long *phigh, char **name);

int
mu_nntp_group (mu_nntp_t nntp, const char *group, unsigned long *total, unsigned long *low, unsigned long *high, char **name)
{
  int status;

  if (nntp == NULL)
    return EINVAL;
  if (group == NULL || *group == '\0')
    return MU_ERR_OUT_PTR_NULL;

  switch (nntp->state)
    {
    case MU_NNTP_NO_STATE:
      status = mu_nntp_writeline (nntp, "GROUP %s\r\n", group);
      MU_NNTP_CHECK_ERROR (nntp, status);
      mu_nntp_debug_cmd (nntp);
      nntp->state = MU_NNTP_GROUP;

    case MU_NNTP_GROUP:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->acknowledge = 0;
      nntp->state = MU_NNTP_GROUP_ACK;

    case MU_NNTP_GROUP_ACK:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE (nntp, MU_NNTP_RESP_CODE_GROUP_SELECTED);
      nntp->state = MU_NNTP_NO_STATE;

      /* parse group.  */
      status = mu_nntp_parse_group (nntp, MU_NNTP_RESP_CODE_GROUP_SELECTED, total, low, high, name);
      MU_NNTP_CHECK_ERROR (nntp, status);
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

static int
mu_nntp_parse_group(mu_nntp_t nntp, int code, unsigned long *ptotal, unsigned long *plow, unsigned long *phigh, char **name)
{
  unsigned long dummy = 0;
  char *buf;
  char format[24];

  if (ptotal == NULL)
    ptotal = &dummy;
  if (plow == NULL)
    plow = &dummy;
  if (phigh == NULL)
    phigh = &dummy;

  /* An nntp response is not longer then 512.  */
  buf = calloc(1, 512);
  if (buf == NULL)
    {
      return ENOMEM;
    }

  sprintf (format, "%d %%d %%d %%d %%%ds", code, 511);
  sscanf (nntp->ack.buf, format, ptotal, plow, phigh, buf);
  if (name)
    {
      *name = buf;
    }
  else
    {
      free (buf);
    }
  return 0;
}
