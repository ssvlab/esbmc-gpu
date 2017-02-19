/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software Foundation,
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

#ifndef _BODY0_H
#define _BODY0_H

#ifdef DMALLOC
#  include <dmalloc.h>
#endif

#include <mailutils/stream.h>
#include <mailutils/body.h>

#include <stdio.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _mu_body
{
  void *owner;
  char *filename;
  mu_stream_t stream;
  mu_stream_t fstream;
  int flags;

  int (*_size)  (mu_body_t, size_t*);
  int (*_lines) (mu_body_t, size_t*);
};

#ifdef __cplusplus
}
#endif

#endif /* _BODY0_H */
