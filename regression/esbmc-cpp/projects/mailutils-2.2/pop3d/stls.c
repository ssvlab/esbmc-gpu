/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "pop3d.h"

/* STLS command -- TLS/SSL encryption */

#ifdef WITH_TLS

int
pop3d_stls (char *arg)
{
  if (strlen (arg) != 0)
    return ERR_BAD_ARGS;

  if (state != initial_state)
    return ERR_WRONG_STATE;

  if (tls_done)
    return ERR_TLS_ACTIVE;

  pop3d_outf ("+OK Begin TLS negotiation\r\n");
  pop3d_flush_output ();

  tls_done = pop3d_init_tls_server ();

  if (!tls_done)
    {
      mu_diag_output (MU_DIAG_ERROR, _("Session terminated"));
      state = ABORT;
      return ERR_UNKNOWN;
    }

  state = AUTHORIZATION;  /* Confirm we're in this state. Necessary for
			     --tls-required to work */

  return OK;
}

#endif /* WITH_TLS */

