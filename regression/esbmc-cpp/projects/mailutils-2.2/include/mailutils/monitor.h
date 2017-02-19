/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2005, 2007, 2010 Free Software Foundation,
   Inc.

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

#ifndef _MAILUTILS_MONITOR_H
#define _MAILUTILS_MONITOR_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mu_monitor
{
  void *data;
  void *owner;
  int allocated;
  int flags;
};
typedef struct mu_monitor *mu_monitor_t;

#define MU_MONITOR_PTHREAD 0
#define MU_MONITOR_INITIALIZER {0, 0, 0, 0}


extern int mu_monitor_create      (mu_monitor_t *, int flags, void *owner);
extern void mu_monitor_destroy    (mu_monitor_t *, void *owner);
extern void *mu_monitor_get_owner (mu_monitor_t);

extern int mu_monitor_rdlock      (mu_monitor_t);
extern int mu_monitor_wrlock      (mu_monitor_t);
extern int mu_monitor_unlock      (mu_monitor_t);
extern int mu_monitor_wait        (mu_monitor_t);
extern int mu_monitor_notify      (mu_monitor_t);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_MONITOR_H */
