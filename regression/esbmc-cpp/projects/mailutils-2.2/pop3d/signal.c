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

#include "pop3d.h"

/* Default signal handler to call the pop3d_abquit() function */

RETSIGTYPE
pop3d_master_signal (int signo)
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
pop3d_child_signal (int signo)
{
  int code;
  
  mu_diag_output (MU_DIAG_CRIT, _("got signal `%s'"), strsignal (signo));

  switch (signo)
    {
    case SIGALRM:
      code = ERR_TIMEOUT;
      break;
      
    case SIGPIPE:
      code = ERR_NO_OFILE;
      break;

    case SIGTERM:
    case SIGHUP:
      code = ERR_TERMINATE;
      break;
      
    default:
      code = ERR_SIGNAL;
    }
  pop3d_abquit (code);
}
