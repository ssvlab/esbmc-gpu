/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
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

#include "guimb.h"

void
util_error (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  fprintf (stderr, "guimb: ");
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  va_end (ap);
}

char *
util_get_sender (int msgno)
{
  mu_header_t header = NULL;
  mu_address_t addr = NULL;
  mu_message_t msg = NULL;
  const char *buffer;
  char *email;

  mu_mailbox_get_message (mbox, msgno, &msg);
  mu_message_get_header (msg, &header);
  if (mu_header_sget_value (header, MU_HEADER_FROM, &buffer)
      || mu_address_create (&addr, buffer))
    {
      mu_envelope_t env = NULL;
      mu_message_get_envelope (msg, &env);
      
      if (mu_envelope_sget_sender (env, &buffer)
	  || mu_address_create (&addr, buffer))
	{
	  util_error (_("cannot determine sender name (msg %d)"), msgno);
	  return NULL;
	}
    }

  if (mu_address_aget_email (addr, 1, &email))
    {
      util_error (_("cannot determine sender name (msg %d)"), msgno);
      mu_address_destroy (&addr);
      return NULL;
    }

  mu_address_destroy (&addr);
  return email;
}

