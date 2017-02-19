/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <syslog.h>
#include <string.h>
#include <mailutils/diag.h>
#include <mailutils/nls.h>
#include <mailutils/errno.h>

const char *mu_program_name;
mu_debug_t mu_diag_debug;

void
mu_set_program_name (const char *name)
{
  const char *progname;

  if (!name)
    progname = name;
  else
    {
      progname = strrchr (name, '/');
      if (progname)
	progname++;
      else
	progname = name;
      
      if (strlen (progname) > 3 && memcmp (progname, "lt-", 3) == 0)
	progname += 3;
    }
  
  mu_program_name = progname;
}

void
mu_diag_init ()
{
  if (!mu_diag_debug)
    {
      int rc = mu_debug_create (&mu_diag_debug, NULL);
      if (rc)
	{
	  fprintf (stderr,
		   _("cannot initialize debug object for diagnostics: %s\n"),
		   mu_strerror (rc));
	  /* That's a fatal error */
	  abort ();
	}
      mu_debug_set_print (mu_diag_debug, mu_diag_stderr_printer, NULL);
    }
}

void
mu_diag_get_debug (mu_debug_t *pdebug)
{
  mu_diag_init ();
  *pdebug = mu_diag_debug;
}

void
mu_diag_set_debug (mu_debug_t debug)
{
  if (mu_diag_debug)
    mu_debug_destroy (&mu_diag_debug, NULL);
  mu_diag_debug = debug;
}

void
mu_diag_vprintf (mu_log_level_t level, const char *fmt, va_list ap)
{
  mu_diag_init ();  
  mu_debug_vprintf (mu_diag_debug, level, fmt, ap);
}

void
mu_diag_printf (mu_log_level_t level, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  mu_diag_vprintf (level, fmt, ap);
  va_end (ap);
}

void
mu_diag_voutput (mu_log_level_t level, const char *fmt, va_list ap)
{
  mu_diag_init ();  
  mu_debug_vprintf (mu_diag_debug, level, fmt, ap);
  mu_debug_printf (mu_diag_debug, level, "\n");
}

void
mu_diag_output (mu_log_level_t level, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  mu_diag_voutput (level, fmt, ap);
  va_end (ap);
}

const char *
mu_diag_level_to_string (mu_log_level_t level)
{
  switch (level)
    {
    case MU_DIAG_EMERG:
      return _("emergency");
      
    case MU_DIAG_ALERT:
      return _("alert");
	
    case MU_DIAG_CRIT:
      return _("critical");
      
    case MU_DIAG_ERROR:
      return _("error");
      
    case MU_DIAG_WARNING:
      return _("warning");
      
    case MU_DIAG_NOTICE:
      return _("notice");
      
    case MU_DIAG_INFO:
      return _("info");
      
    case MU_DIAG_DEBUG:
      return _("debug");
    }
  return _("unknown");
}

int
mu_diag_stderr_printer (void *data, mu_log_level_t level, const char *buf)
{
  if (mu_program_name)
    fprintf (stderr, "%s: ", mu_program_name);
  if (level != MU_DIAG_ERROR)
    fprintf (stderr, "%s: ", mu_diag_level_to_string (level));
  fputs (buf, stderr);
  return 0;
}

void
mu_diag_funcall (mu_log_level_t level, const char *func,
		 const char *arg, int err)
{
  if (err)
    /* TRANSLATORS: First %s stands for function name, second for its
       arguments, third one for the actual error message. */
    mu_diag_output (level, _("%s(%s) failed: %s"), func, arg ? arg : "",
		    mu_strerror (err));
  else
    /* TRANSLATORS: First %s stands for function name, second for its
       arguments. */
    mu_diag_output (level, _("%s(%s) failed"), func, arg ? arg : "");
}
