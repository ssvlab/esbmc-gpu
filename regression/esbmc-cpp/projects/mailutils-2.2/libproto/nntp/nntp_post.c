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
mu_nntp_post (mu_nntp_t nntp, mu_stream_t stream)
{
  int status;

  if (nntp == NULL)
    return EINVAL;

  switch (nntp->state)
    {
    case MU_NNTP_NO_STATE:
      status = mu_nntp_writeline (nntp, "POST\r\n");
      MU_NNTP_CHECK_ERROR (nntp, status);
      mu_nntp_debug_cmd (nntp);
      nntp->state = MU_NNTP_POST;

    case MU_NNTP_POST:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->acknowledge = 0;
      nntp->state = MU_NNTP_POST_ACK;

    case MU_NNTP_POST_ACK:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE (nntp, MU_NNTP_RESP_CODE_SEND_ARTICLE);
      if (nntp->post.buf != NULL)
	{
	  free (nntp->post.buf);
	}
      nntp->post.buf = calloc (1, 256);
      if (nntp->post.buf == NULL)
	{
	  MU_NNTP_CHECK_ERROR (nntp, ENOMEM);
	}
      nntp->post.len = 256;
      nntp->post.offset = 0;
      nntp->post.nread = 0;
      nntp->post.sent_crlf = 0;
      nntp->state = MU_NNTP_POST_0;

    post_loop:
    case MU_NNTP_POST_0:
      status = mu_stream_readline (stream, nntp->post.buf, nntp->post.len, nntp->post.offset, &(nntp->post.nread));
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->post.offset += nntp->post.nread;
      if  (nntp->post.nread > 0)
	{
	  if (nntp->post.buf[nntp->post.nread - 1] == '\n')
	    {
	      nntp->post.buf[nntp->post.nread - 1] = '\0';
	      if (nntp->post.sent_crlf && nntp->post.buf[0] == '.')
		{
		  status = mu_nntp_writeline (nntp, ".%s\r\n", nntp->post.buf);
		}
	      else
		{
		  status = mu_nntp_writeline (nntp, "%s\r\n", nntp->post.buf);
		}
	      nntp->post.sent_crlf = 1;
	    }
	  else
	    {
	      if (nntp->post.sent_crlf && nntp->post.buf[0] == '.')
		{
		  status = mu_nntp_writeline (nntp, ".%s", nntp->post.buf);
		}
	      else
		{
		  status = mu_nntp_writeline (nntp, "%s", nntp->post.buf);
		}
	      nntp->post.sent_crlf = 0;
	    }
	  MU_NNTP_CHECK_ERROR (nntp, status);
	}
      nntp->state = MU_NNTP_POST_1;

    case MU_NNTP_POST_1:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      if (nntp->post.nread > 0)
	{
	  goto post_loop;
	}
      if (nntp->post.sent_crlf)
	status = mu_nntp_writeline (nntp, ".\r\n");
      else
	status = mu_nntp_writeline (nntp, "\r\n.\r\n");
      if (nntp->post.buf != NULL)
	{
	  free (nntp->post.buf);
	  nntp->post.buf = NULL;
	  nntp->post.len = 0;
	  nntp->post.offset = 0;
	  nntp->post.nread = 0;
	  nntp->post.sent_crlf = 0;
	}
      MU_NNTP_CHECK_ERROR (nntp, status);
      nntp->state = MU_NNTP_POST_2;

    case MU_NNTP_POST_2:
      status = mu_nntp_send (nntp);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      nntp->state = MU_NNTP_POST_3;

    case MU_NNTP_POST_3:
      status = mu_nntp_response (nntp, NULL, 0, NULL);
      MU_NNTP_CHECK_EAGAIN (nntp, status);
      mu_nntp_debug_ack (nntp);
      MU_NNTP_CHECK_CODE (nntp, MU_NNTP_RESP_CODE_ARTICLE_RECEIVED);
      nntp->state = MU_NNTP_NO_STATE;
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
