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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <syslog.h>
#include <string.h>
#include <mailutils/error.h>


/* Historic shortcuts for mu_diag_ functions */

int
mu_verror (const char *fmt, va_list ap)
{
  mu_diag_voutput (MU_DIAG_ERROR, fmt, ap);
  return 0;
}

int
mu_error (const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  mu_verror (fmt, ap);
  va_end (ap);
  return 0;
}


/* Compatibility layer */
int
mu_default_error_printer (const char *fmt, va_list ap)
{
  if (mu_program_name)
    fprintf (stderr, "%s: ", mu_program_name);
  vfprintf (stderr, fmt, ap);
  fputc ('\n', stderr);
  return 0;
}

int
mu_syslog_error_printer (const char *fmt, va_list ap)
{
#ifdef HAVE_VSYSLOG
  vsyslog (LOG_CRIT, fmt, ap);
#else
  char buf[128];
  vsnprintf (buf, sizeof buf, fmt, ap);
  syslog (LOG_CRIT, "%s", buf);
#endif
  return 0;
}

static void
compat_error_printer0 (mu_error_pfn_t pfn, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  pfn (fmt, ap);
  va_end (ap);
}

static int
compat_error_printer (void *data, mu_log_level_t level, const char *buf)
{
  if (!data)
    mu_diag_stderr_printer (NULL, level, buf);
  else
    {
      int len = strlen (buf);
      if (len > 0 && buf[len-1] == '\n')
	len--;
      compat_error_printer0 (data, "%-.*s", len, buf);
    }
  return 0;
}

void
mu_error_set_print (mu_error_pfn_t pfn)
{
  mu_debug_t debug;
  mu_diag_get_debug (&debug);
  mu_debug_set_print (debug, compat_error_printer, NULL);
  mu_debug_set_data (debug, pfn, NULL, NULL);
#if 0
 {
   static int warned;
   if (!warned)
     {
       warned = 1;
       mu_diag_output ("this program uses mu_error_set_print, which is deprecated");
     }
#endif
}


