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
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <lbuf.h>

struct _line_buffer {
  char *buffer;        /* Line buffer */
  size_t size;         /* Allocated size */
  size_t level;        /* Current filling level */
};

int
_auth_lb_create (struct _line_buffer **s)
{
  *s = malloc (sizeof (**s));
  if (!*s)
    return ENOMEM;
  (*s)->buffer = NULL;
  (*s)->size = 0;
  (*s)->level = 0;
  return 0;
}

void
_auth_lb_destroy (struct _line_buffer **s)
{
  if (s && *s)
    {
      free ((*s)->buffer);
      free (*s);
      *s = NULL;
    }
}

void
_auth_lb_drop (struct _line_buffer *s)
{
  s->level = 0;
}

int
_auth_lb_grow (struct _line_buffer *s, const char *ptr, size_t size)
{
  if (!s->buffer)
    {
      s->buffer = malloc (size);
      s->size = size;
      s->level = 0;
    }
  else if (s->size - s->level < size)
    {
      size_t newsize = s->size + size;
      s->buffer = realloc (s->buffer, newsize);
      if (s->buffer)
	s->size = newsize;
    }

  if (!s->buffer)
    return ENOMEM;
  
  memcpy (s->buffer + s->level, ptr, size);
  s->level += size;
  return 0;
}

int
_auth_lb_read (struct _line_buffer *s, char *optr, size_t osize)
{
  int len;

  len = s->level > osize ? osize : s->level;
  memcpy (optr, s->buffer, len);
  if (s->level > len)
    {
      memmove (s->buffer, s->buffer + len, s->level - len);
      s->level -= len;
    }
  else if (s->level == len)
    s->level = 0;
    
  return len;
}

int
_auth_lb_readline (struct _line_buffer *s, char *ptr, size_t size)
{
  char *p = strchr (s->buffer, '\n');

  if (p && p - s->buffer + 1 < size)
    size = p - s->buffer + 1;
  return _auth_lb_read (s, ptr, size);
}

int
_auth_lb_writelines (struct _line_buffer *s, const char *iptr, size_t isize,
		     off_t offset,
		     int (*wr) (void *data, char *start, char *end),
		     void *data,
		     size_t *nbytes)
{
  if (s->level > 2)
    {
      char *start, *end;
      
      for (start = s->buffer,
		   end = memchr (start, '\n', s->buffer + s->level - start);
	   end && end < s->buffer + s->level;
	   start = end + 1,
		   end = memchr (start, '\n', s->buffer + s->level - start))
	if (end[-1] == '\r')
	  {
	    int rc = wr (data, start, end);
	    if (rc)
	      return rc;
	  }

      if (start > s->buffer)
	{
	  if (start < s->buffer + s->level)
	    {
	      int rest = s->buffer + s->level - start;
	      memmove (s->buffer, start, rest);
	      s->level = rest;
	    }
	  else 
	    s->level = 0;
	}
    }

  if (nbytes)
    *nbytes = isize;
  return 0;
}

int
_auth_lb_level (struct _line_buffer *s)
{
  return s->level;
}

char *
_auth_lb_data (struct _line_buffer *s)
{
  return s->buffer;
}
