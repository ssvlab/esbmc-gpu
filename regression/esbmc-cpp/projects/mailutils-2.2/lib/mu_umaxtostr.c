/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2010 Free Software Foundation, Inc.

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

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mu_umaxtostr.h>

static char **buffer_pool;
static size_t buffer_size;
#define BUFFER_SIZE_INIT 4
#define BUFFER_SIZE_INCR 4

static char *
get_buffer (unsigned slot)
{
  if (!buffer_pool)
    {
      buffer_size = BUFFER_SIZE_INIT;
      buffer_pool = calloc (buffer_size, sizeof *buffer_pool);
    }
  else if (slot >= buffer_size)
    {
      buffer_size += (slot + BUFFER_SIZE_INCR - 1) / BUFFER_SIZE_INCR;
      buffer_pool = realloc (buffer_pool, buffer_size * sizeof *buffer_pool);
    }
  if (!buffer_pool)
    return NULL;
  if (buffer_pool[slot] == NULL)
    buffer_pool[slot] = malloc (UINTMAX_STRSIZE_BOUND);
  return buffer_pool[slot];
}

const char *
mu_umaxtostr (unsigned slot, uintmax_t val)
{
  char *s = get_buffer (slot);
  if (!s)
    return mu_strerror(ENOMEM);
  return umaxtostr (val, s);
}
