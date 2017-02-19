/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <signal.h>

void
mu_set_signals (RETSIGTYPE (*handler) (int signo), int *sigv, int sigc)
{
  int i;
  struct sigaction act;

  act.sa_flags = 0;
  sigemptyset (&act.sa_mask);
  for (i = 0; i < sigc; i++)
    sigaddset (&act.sa_mask, i);
      
  for (i = 0; i < sigc; i++)
    {
      act.sa_handler = handler;
      sigaction (sigv[i], &act, NULL);
    }
}
