/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2008, 2009, 2010 Free Software Foundation,
   Inc.

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

#include "imap4d.h"

#ifdef WITH_TLS

static int tls_available;
static int tls_done;

/*
6.2.1.  STARTTLS Command

   Arguments:  none

   Responses:  no specific response for this command

   Result:     OK - starttls completed, begin TLS negotiation
               BAD - command unknown or arguments invalid
*/
int
imap4d_starttls (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int status;

  if (!tls_available || tls_done)
    return util_finish (command, RESP_BAD, "Invalid command");

  if (imap4d_tokbuf_argc (tok) != 2)
    return util_finish (command, RESP_BAD, "Invalid arguments");

  util_atexit (mu_deinit_tls_libs);

  status = util_finish (command, RESP_OK, "Begin TLS negotiation");
  util_flush_output ();
  tls_done = imap4d_init_tls_server ();

  if (tls_done)
    {
      imap4d_capability_remove (IMAP_CAPA_STARTTLS);
      
      login_disabled = 0;
      imap4d_capability_remove (IMAP_CAPA_LOGINDISABLED);

      tls_required = 0;
      imap4d_capability_remove (IMAP_CAPA_XTLSREQUIRED);
    }
  else
    {
      mu_diag_output (MU_DIAG_ERROR, _("session terminated"));
      util_bye ();
      exit (EX_OK);
    }

  return status;
}

void
starttls_init ()
{
  tls_available = mu_check_tls_environment ();
  if (tls_available)
    tls_available = mu_init_tls_libs ();
  if (tls_available)
    imap4d_capability_add (IMAP_CAPA_STARTTLS);
}

#endif /* WITH_TLS */

/* EOF */
