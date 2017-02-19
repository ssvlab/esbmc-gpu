/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2008, 2010 Free
   Software Foundation, Inc.

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

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <debug0.h>

mu_debug_printer_fp mu_debug_default_printer = mu_debug_stderr_printer;

int
mu_debug_create (mu_debug_t *pdebug, void *owner)
{
  mu_debug_t debug;
  if (pdebug == NULL)
    return MU_ERR_OUT_PTR_NULL;
  debug = calloc (sizeof (*debug), 1);
  if (debug == NULL)
    return ENOMEM;
  debug->printer = NULL;
  debug->owner = owner;
  *pdebug = debug;
  return 0;
}

void
mu_debug_destroy (mu_debug_t *pdebug, void *owner)
{
  if (pdebug && *pdebug)
    {
      mu_debug_t debug = *pdebug;
      if (debug->owner == owner)
	{
	  mu_off_t len = 0;
	  int rc = mu_stream_size (debug->stream, &len);
	  if (rc == 0 && len)
	    /* Flush leftover data */
	    mu_debug_printf (debug, 0, "\n");

	  mu_stream_destroy (&debug->stream,
			     mu_stream_get_owner (debug->stream));
	  if (debug->destroy)
	    debug->destroy (debug->data);
	  free (*pdebug);
	  *pdebug = NULL;
	}
    }
}

void *
mu_debug_get_owner (mu_debug_t debug)
{
  return (debug) ? debug->owner : NULL;
}

int
mu_debug_set_level (mu_debug_t debug, mu_log_level_t level)
{
  if (debug == NULL)
    return EINVAL;
  debug->level = level;
  return 0;
}

int
mu_debug_get_level (mu_debug_t debug, mu_log_level_t *plevel)
{
  if (debug == NULL)
    return EINVAL;
  if (plevel)
    *plevel = debug->level;
  return 0;
}

int
mu_debug_set_print (mu_debug_t debug, mu_debug_printer_fp printer, void *owner)
{
  if (debug == NULL)
    return EINVAL;
  if (debug->owner != owner)
    return EACCES;
  debug->printer = printer;
  return 0;
}

int
mu_debug_set_data (mu_debug_t debug, void *data, void (*destroy) (void*),
		   void *owner)
{
  if (!debug)
    return EINVAL;
  if (debug->owner != owner)
    return EACCES;
  debug->data = data;
  debug->destroy = destroy;
  return 0;
}

static void
debug_format_prefix (mu_debug_t debug)
{
  int need_space = 0;
  if (debug->locus.file)
    {
      mu_stream_sequential_printf (debug->stream, "%s:%d:",
				   debug->locus.file, debug->locus.line);
      need_space = 1;
    }
  
  if (debug->function)
    {
      mu_stream_sequential_printf (debug->stream, "%s:",
				   debug->function);
      need_space = 1;
    }
  
  if (need_space)
    mu_stream_sequential_write (debug->stream, " ", 1);
}

int
mu_debug_vprintf (mu_debug_t debug, mu_log_level_t level,
		  const char *format, va_list ap)
{
  mu_debug_printer_fp printer;
  
  if (debug == NULL || format == NULL)
    return EINVAL;

  printer = debug->printer ? debug->printer : mu_debug_default_printer;
  if (printer)
    {
      mu_off_t len;
      mu_transport_t tbuf;
      char *ptr, *start, *p;
      size_t nseg;
      
      if (debug->stream == NULL)
	{
	  int rc = mu_memory_stream_create (&debug->stream, NULL, 0);
	  if (rc)
	    {
	      fprintf (stderr,
		       _("cannot create memory stream for debugging "
			 "output: %s\n"),
		       mu_strerror (rc));
	      vfprintf (stderr, format, ap);
	      return rc;
	    }
	}

      if (mu_stream_size (debug->stream, &len) == 0 && len == 0)
	debug_format_prefix (debug);
      
      mu_stream_sequential_vprintf (debug->stream, format, ap);
      
      mu_stream_get_transport (debug->stream, &tbuf);
      start = (char*) tbuf;
      mu_stream_size (debug->stream, &len);
      ptr = start;
      nseg = 0;
      for (p = ptr = start; p < start + len; p++)
	{
	  if (*p == '\n')
	    {
	      int c = *++p;
	      *p = 0;
	      printer (debug->data, level, ptr);
	      *p = c;
	      ptr = p;
	      nseg++;
	    }
	}

      if (nseg)
	{
	  if (start[len - 1] != '\n')
	    {
	      size_t s = len - (ptr - start);
	      memmove (start, ptr, s);
	    }
	  else
	    len = 0;

	  mu_stream_truncate (debug->stream, len);
	  mu_stream_seek (debug->stream, len, SEEK_SET);
	}
    }
  else
    vfprintf (stderr, format, ap);

  return 0;
}

int
mu_debug_printf (mu_debug_t debug, mu_log_level_t level,
		 const char *format, ...)
{
  va_list ap;

  va_start (ap, format);
  mu_debug_vprintf (debug, level, format, ap);
  va_end (ap);
  return 0;
}
  

int
mu_debug_print (mu_debug_t debug, mu_log_level_t level,
		const char *format, ...)
{
  va_list ap;
  mu_debug_printv (debug, level, format, ap);
  va_end (ap);
  return 0;
}

int
mu_debug_printv (mu_debug_t debug, mu_log_level_t level, const char *format,
		 va_list ap)
{
  if (debug == NULL || format == NULL)
    return EINVAL;
  if (debug->level & MU_DEBUG_LEVEL_MASK (level))
    return 0;
  return mu_debug_vprintf (debug, level, format, ap);
}

int
mu_debug_check_level (mu_debug_t debug, mu_log_level_t level)
{
  if (!debug)
    return 0;
  return debug->level & MU_DEBUG_LEVEL_MASK (level);
}

int
mu_debug_set_locus (mu_debug_t debug, const char *file, int line)
{
  if (!debug)
    return EINVAL;
  debug->locus.file = file;
  debug->locus.line = line;
  return 0;
}

int
mu_debug_get_locus (mu_debug_t debug, struct mu_debug_locus *ploc)
{
  if (!debug)
    return EINVAL;
  *ploc = debug->locus;
  return 0;
}

int
mu_debug_set_function (mu_debug_t debug, const char *function)
{
  if (!debug)
    return EINVAL;
  debug->function = function;
  return 0;
}

int
mu_debug_get_function (mu_debug_t debug, const char **pfunction)
{
  if (!debug)
    return EINVAL;
  *pfunction = debug->function;
  return 0;
}

