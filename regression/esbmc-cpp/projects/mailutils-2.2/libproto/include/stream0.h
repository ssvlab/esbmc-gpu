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

#ifndef _STREAM0_H
#define _STREAM0_H

#include <mailutils/stream.h>
#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Read buffer */
struct rbuffer
{
  char *base;
  char *ptr;
  size_t count;
  size_t bufsiz;
  mu_off_t offset;
};

struct _mu_stream
{
  void *owner;
  mu_property_t property;

  int flags;
  int state;

  /* Read space */
  struct rbuffer rbuffer;
  
  /* Stream pointer for sequential offset.  */
  mu_off_t offset;
  
  void (*_destroy) (mu_stream_t);
  int (*_open)     (mu_stream_t);
  int (*_close)    (mu_stream_t);
  int (*_get_transport2) (mu_stream_t, mu_transport_t *, mu_transport_t *);
  int (*_read)     (mu_stream_t, char *, size_t, mu_off_t, size_t *);
  int (*_readline) (mu_stream_t, char *, size_t, mu_off_t, size_t *);
  int (*_write)    (mu_stream_t, const char *, size_t, mu_off_t, size_t *);
  int (*_truncate) (mu_stream_t, mu_off_t);
  int (*_size)     (mu_stream_t, mu_off_t *);
  int (*_flush)    (mu_stream_t);
  int (*_setbufsiz)(mu_stream_t, size_t);
  int (*_strerror) (mu_stream_t, const char **);
  int (*_wait)     (mu_stream_t, int *pflags, struct timeval *tvp);
  int (*_shutdown) (mu_stream_t, int how);
};

#ifdef __cplusplus
}
#endif

#endif /* _STREAM0_H */
