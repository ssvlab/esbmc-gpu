/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2010 Free
   Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

/* This file implements an MH draftfile stream: a read-only stream used
   to transparently pass MH draftfiles to mailers. The only difference
   between the usual RFC822 and MH draft is that the latter allows to use
   a string of dashes to separate the headers from the body. */

#include <mh.h>
#include <mailutils/stream.h>

mu_message_t
mh_stream_to_message (mu_stream_t instream)
{
  int rc;
  mu_message_t msg;

  rc = mu_stream_to_message (instream, &msg);
  if (rc)
    {
      mu_error (_("cannot open draft message stream: %s"),
		mu_strerror (rc));
      return NULL;
    }
  return msg;
}
