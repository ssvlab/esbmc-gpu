/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2010 Free Software Foundation, Inc.

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
#include <mailutils/sys/pop3.h>

int
mu_pop3_set_timeout (mu_pop3_t pop3, int timeout)
{
  /* Sanity checks.  */
  if (pop3 == NULL)
    return EINVAL;

  pop3->timeout = timeout;
  return 0;
}

int
mu_pop3_get_timeout (mu_pop3_t pop3, int *ptimeout)
{
  /* Sanity checks.  */
  if (pop3 == NULL)
    return EINVAL;
  if (ptimeout == NULL)
    return MU_ERR_OUT_PTR_NULL;

  *ptimeout = pop3->timeout;
  return 0;
}
