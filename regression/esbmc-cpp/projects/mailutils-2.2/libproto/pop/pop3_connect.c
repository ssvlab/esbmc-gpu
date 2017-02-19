/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2009, 2010 Free Software Foundation, Inc.

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

#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <mailutils/sys/pop3.h>

static int mu_pop3_sleep (int seconds);

/* Open the connection to the server. The server sends an affirmative greeting
   that may contain a timestamp for APOP.  */
int
mu_pop3_connect (mu_pop3_t pop3)
{
  int status;

  /* Sanity checks.  */
  if (pop3 == NULL)
    return EINVAL;

  /* A networking stack.  */
  if (pop3->carrier == NULL)
    return EINVAL;

  /* Enter the pop state machine, and boogy: AUTHORISATION State.  */
  switch (pop3->state)
    {
    default:
      /* FALLTHROUGH */
      /* If pop3 was in an error state going through here should clear it.  */

    case MU_POP3_NO_STATE:
      /* If the stream was previoulsy open this is sudden death:
         for many pop servers, it is important to let them time to remove any locks or move
         the .user.pop files.  This happen when we do close() and immediately open().
         For example, the user does not want to read the entire file, and wants to start
	 to read a new message, closing the connection and immediately
	 contacting the server again, and he'll end up having
	 "-ERR Mail Lock busy" or something similar. To prevent this race
	 condition we sleep 2 seconds.  You can see this behaviour in an
	 environment where QPopper(Qualcomm POP3 server) is use and the user as a big mailbox. */
      status = mu_pop3_disconnect (pop3);
      if (status != 0)
        mu_pop3_sleep (2);
      pop3->state = MU_POP3_CONNECT;

    case MU_POP3_CONNECT:
      /* Establish the connection.  */
      status = mu_stream_open (pop3->carrier);
      MU_POP3_CHECK_EAGAIN (pop3, status);
      pop3->acknowledge = 0;
      pop3->state = MU_POP3_GREETINGS;

    case MU_POP3_GREETINGS:
      /* Get the greetings.  */
      {
	size_t len = 0;
	char *right, *left;
	status = mu_pop3_response (pop3, NULL, 0, &len);
	MU_POP3_CHECK_EAGAIN (pop3, status);
	mu_pop3_debug_ack (pop3);
	if (mu_c_strncasecmp (pop3->ack.buf, "+OK", 3) != 0)
	  {
	    mu_stream_close (pop3->carrier);
	    pop3->state = MU_POP3_NO_STATE;
	    return EACCES;
	  }

	/* Get the timestamp.  */
	right = memchr (pop3->ack.buf, '<', len);
	if (right)
	  {
	    len = len - (right - pop3->ack.buf);
	    left = memchr (right, '>', len);
	    if (left)
	      {
		len = left - right + 1;
		pop3->timestamp = calloc (len + 1, 1);
		if (pop3->timestamp == NULL)
		  {
		    mu_stream_close (pop3->carrier);
		    MU_POP3_CHECK_ERROR (pop3, ENOMEM);
		  }
		memcpy (pop3->timestamp, right, len);
	      }
	  }
	pop3->state = MU_POP3_NO_STATE;
      }
    } /* End AUTHORISATION state. */

  return status;
}

/* GRRRRR!!  We can not use sleep in the library since this we'll
   muck up any alarm() done by the user.  */
static int
mu_pop3_sleep (int seconds)
{
  struct timeval tval;
  tval.tv_sec = seconds;
  tval.tv_usec = 0;
  return select (1, NULL, NULL, NULL, &tval);
}
