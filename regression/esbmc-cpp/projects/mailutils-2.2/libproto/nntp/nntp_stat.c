/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2007, 2010 Free Software
   Foundation, Inc.

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

int
mu_nntp_stat (mu_nntp_t nntp, unsigned long number, unsigned long *pnumber, char **mid)
{
  int status;
  char *message_id = NULL;
  if (number != 0)
    {
      message_id = malloc (128);
      if (message_id == NULL)
	{
	  return ENOMEM;
	}
      snprintf (message_id, 127, "%ld", number);
    }
  status = mu_nntp_stat_id (nntp, message_id, pnumber, mid);
  if (message_id)
    {
      free (message_id);
    }
  return status;
}

int
mu_nntp_stat_id (mu_nntp_t nntp, const char *message_id, unsigned long *number, char **mid)
{
  int status;

  if (nntp == NULL)
    return EINVAL;

  switch (nntp->state)
    {
    case MU_NNTP_NO_STATE:
      if (message_id == NULL || *message_id == '\0')
	{
	  status = mu_nntp_writeline (nntp, "STAT\r\n");
	}
      else
	{
	  status = mu_nntp_writeline (nntp, "STAT %s\r\n", message_id);
	}
      MU_NNTP_CHECK_ERROR (nntp, status);
      mu_nntp_debug_cmd (nntp);
      nntp->state = MU_NNTP_STAT;

    case MU_NNTP_STAT:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->acknowledge = 0;
      nntp->state = MU_NNTP_STAT_ACK;

    case MU_NNTP_STAT_ACK:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE(nntp, MU_NNTP_RESP_CODE_ARTICLE_FOUND);
      nntp->state = MU_NNTP_NO_STATE;

      /* parse the answer now. */
      status = mu_nntp_parse_article(nntp, MU_NNTP_RESP_CODE_ARTICLE_FOUND, number, mid);
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
