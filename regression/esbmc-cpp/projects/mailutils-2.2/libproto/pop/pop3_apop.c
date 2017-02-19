/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2005, 2007, 2010 Free Software
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

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <mailutils/sys/pop3.h>
#include <mailutils/md5.h>

/*
 * APOP name digest
 *  a string identifying a mailbox and a MD5 digest string (both required)
 */
int
mu_pop3_apop (mu_pop3_t pop3, const char *user, const char *secret)
{
  int status;

  /* Sanity checks.  */
  if (pop3 == NULL || user == NULL || secret == NULL)
    {
      return EINVAL;
    }

  /* The server did not offer a timestamp in the greeting, bailout early.  */
  if (pop3->timestamp == NULL)
    {
      return ENOTSUP;
    }

  switch (pop3->state)
    {
      /* Generate the md5 from the secret and timestamp.  */
    case MU_POP3_NO_STATE:
      {
	struct mu_md5_ctx md5context;
	unsigned char md5digest[16];
	char digest[64]; /* Really it just has to be 32 + 1(null).  */
	char *tmp;
	size_t n;

	mu_md5_init_ctx (&md5context);
	mu_md5_process_bytes (pop3->timestamp, strlen (pop3->timestamp), &md5context);
	mu_md5_process_bytes (secret, strlen (secret), &md5context);
	mu_md5_finish_ctx (&md5context, md5digest);
	for (tmp = digest, n = 0; n < 16; n++, tmp += 2)
	  {
	    sprintf (tmp, "%02x", md5digest[n]);
	  }
	*tmp = '\0';

	status = mu_pop3_writeline (pop3, "APOP %s %s\r\n", user, digest);
	/* Obscure the digest, for security reasons.  */
	memset (digest, '\0', sizeof digest);
	MU_POP3_CHECK_ERROR (pop3, status);
	mu_pop3_debug_cmd (pop3);
	pop3->state = MU_POP3_APOP;
      }

    case MU_POP3_APOP:
      status = mu_pop3_send (pop3);
      MU_POP3_CHECK_EAGAIN (pop3, status);
      /* Obscure the digest, for security reasons.  */
      memset (pop3->io.buf, '\0', pop3->io.len);
      pop3->acknowledge = 0;
      pop3->state = MU_POP3_APOP_ACK;

    case MU_POP3_APOP_ACK:
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

      /* No case in the switch another operation was in progress.  */
    default:
      status = EINPROGRESS;
    }

  return status;
}
