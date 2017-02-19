/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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

#ifndef _FILTER0_H
#define _FILTER0_H

#include <mailutils/filter.h>
#include <mailutils/list.h>
#include <mailutils/monitor.h>
#include <mailutils/property.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _mu_filter
{
  mu_stream_t stream;
  mu_stream_t filter_stream;
  mu_property_t property;
  int direction;
  int type;
  void *data;
  int  (*_read)     (mu_filter_t, char *, size_t, mu_off_t, size_t *);
  int  (*_readline) (mu_filter_t, char *, size_t, mu_off_t, size_t *);
  int  (*_write)    (mu_filter_t, const char *, size_t, mu_off_t, size_t *);
  void (*_destroy)  (mu_filter_t);
};

#ifdef __cplusplus
}
#endif

#endif /* _FILTER0_H */
