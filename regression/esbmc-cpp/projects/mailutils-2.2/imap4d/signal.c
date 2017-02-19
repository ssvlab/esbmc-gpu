/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2007, 2008, 2009, 2010 Free
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

#define __USE_MISC
#include "imap4d.h"

/* Default signal handler to call the imap4d_bye() function */

RETSIGTYPE
imap4d_master_signal (int signo)
{
  int code;

  mu_diag_output (MU_DIAG_CRIT, _("MASTER: exiting on signal (%s)"),
		  strsignal (signo));
  switch (signo)
    {
    case SIGTERM:
    case SIGHUP:
    case SIGQUIT:
    case SIGINT:
      code = EX_OK;
      break;

    default:
      code = EX_SOFTWARE;
      break;
    }

  exit (code); 
}

RETSIGTYPE
imap4d_child_signal (int signo)
{
  imap4d_child_signal_setup (SIG_IGN);
  mu_diag_output (MU_DIAG_CRIT, _("got signal `%s'"), strsignal (signo));
  switch (signo)
    {
    case SIGTERM:
    case SIGHUP:
      signo = ERR_TERMINATE;
      break;
      
    case SIGALRM:
      signo = ERR_TIMEOUT;
      break;
      
    case SIGPIPE:
      signo = ERR_NO_OFILE;
      break;
      
    default:
      signo = ERR_SIGNAL;
    }
  imap4d_bye (signo);
}
