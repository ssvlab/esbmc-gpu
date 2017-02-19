/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2007, 2010 Free Software Foundation, Inc.

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

#include <stdlib.h>
#include <errno.h>
#include <mailutils/sys/nntp.h>

int
mu_nntp_set_debug (mu_nntp_t nntp, mu_debug_t debug)
{
  if (nntp == NULL)
    return EINVAL;
  if (nntp->debug)
    mu_debug_destroy (&nntp->debug, NULL);
  nntp->debug = debug;
  return 0;
}

/* FIXME: should be a macro */
int
mu_nntp_debug_cmd (mu_nntp_t nntp)
{
  MU_DEBUG (nntp->debug, MU_DEBUG_PROT, nntp->io.buf);
  return 0;
}

int
mu_nntp_debug_ack (mu_nntp_t nntp)
{
  MU_DEBUG1 (nntp->debug, MU_DEBUG_PROT, "%s\n", nntp->ack.buf);
  return 0;
}
