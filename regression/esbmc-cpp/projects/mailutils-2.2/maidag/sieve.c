/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "maidag.h"

static int
_sieve_debug_printer (void *unused, const char *fmt, va_list ap)
{
  mu_diag_vprintf (MU_DIAG_DEBUG, fmt, ap);
  return 0;
}

static void
_sieve_action_log (void *user_name,
		   const mu_sieve_locus_t *locus, size_t msgno,
		   mu_message_t msg,
		   const char *action, const char *fmt, va_list ap)
{
  int pfx = 0;
  mu_debug_t debug;

  mu_diag_get_debug (&debug);
  mu_debug_set_locus (debug, locus->source_file, locus->source_line);
  
  mu_diag_printf (MU_DIAG_NOTICE, _("(user %s) "), (char*) user_name);
  if (message_id_header)
    {
      mu_header_t hdr = NULL;
      char *val = NULL;
      mu_message_get_header (msg, &hdr);
      if (mu_header_aget_value (hdr, message_id_header, &val) == 0
	  || mu_header_aget_value (hdr, MU_HEADER_MESSAGE_ID, &val) == 0)
	{
	  pfx = 1;
	  mu_diag_printf (MU_DIAG_NOTICE, _("%s on msg %s"), action, val);
	  free (val);
	}
    }
  
  if (!pfx)
    {
      size_t uid = 0;
      mu_message_get_uid (msg, &uid);
      mu_diag_printf (MU_DIAG_NOTICE, _("%s on msg uid %lu"), action,
		      (unsigned long) uid);
    }
  
  if (fmt && strlen (fmt))
    {
      mu_diag_printf (MU_DIAG_NOTICE, "; ");
      mu_diag_vprintf (MU_DIAG_NOTICE, fmt, ap);
    }
  mu_diag_printf (MU_DIAG_NOTICE, "\n");
  mu_debug_set_locus (debug, NULL, 0);
}

static int
_sieve_parse_error (void *user_name, const char *filename, int lineno,
		    const char *fmt, va_list ap)
{
  mu_debug_t debug;

  mu_diag_get_debug (&debug);
  if (filename)
    mu_debug_set_locus (debug, filename, lineno);

  mu_diag_printf (MU_DIAG_ERROR, _("(user %s) "), (char*) user_name);
  mu_diag_vprintf (MU_DIAG_ERROR, fmt, ap);
  mu_diag_printf (MU_DIAG_ERROR, "\n");
  mu_debug_set_locus (debug, NULL, 0);
  return 0;
}

int
sieve_check_msg (mu_message_t msg, struct mu_auth_data *auth, const char *prog)
{
  int rc;
  mu_sieve_machine_t mach;

  rc = mu_sieve_machine_init (&mach, auth->name);
  if (rc)
    {
      mu_error (_("Cannot initialize sieve machine: %s"),
		mu_strerror (rc));
    }
  else
    {
      mu_sieve_set_debug (mach, _sieve_debug_printer);
      mu_sieve_set_debug_level (mach, sieve_debug_flags);
      mu_sieve_set_parse_error (mach, _sieve_parse_error);
      if (sieve_enable_log)
	mu_sieve_set_logger (mach, _sieve_action_log);
	  
      rc = mu_sieve_compile (mach, prog);
      if (rc == 0)
	mu_sieve_message (mach, msg);
    }
  return 0;
}

