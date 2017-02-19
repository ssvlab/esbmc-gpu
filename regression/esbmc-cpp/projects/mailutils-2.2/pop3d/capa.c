/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2003, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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

/*
  The CAPA Command

  The POP3 CAPA command returns a list of capabilities supported by the
  POP3 server.  It is available in both the AUTHORIZATION and
  TRANSACTION states.

  Capabilities available in the AUTHORIZATION state MUST be announced
  in both states.  */
int
pop3d_capa (char *arg)
{
  if (strlen (arg) != 0)
    return ERR_BAD_ARGS;

  if (state != initial_state && state != TRANSACTION)
    return ERR_WRONG_STATE;

  pop3d_outf ("+OK Capability list follows\r\n");
  pop3d_outf ("TOP\r\n");
  pop3d_outf ("USER\r\n");
  pop3d_outf ("UIDL\r\n");
  pop3d_outf ("RESP-CODES\r\n");
  pop3d_outf ("PIPELINING\r\n");

#ifdef WITH_TLS
  if (tls_available && tls_done == 0)
    pop3d_outf ("STLS\r\n");
#endif /* WITH_TLS */

  login_delay_capa ();
  /* This can be implemented by setting an header field on the message.  */
  if (expire == EXPIRE_NEVER)
    pop3d_outf ("EXPIRE NEVER\r\n");
  else 
    pop3d_outf ("EXPIRE %s\r\n", mu_umaxtostr (0, expire));

  if (state == INITIAL)
    pop3d_outf ("XTLSREQUIRED\r\n");
  
  if (state == TRANSACTION)	/* let's not advertise to just anyone */
    pop3d_outf ("IMPLEMENTATION %s\r\n", PACKAGE_STRING);
  pop3d_outf (".\r\n");
  return OK;
}
