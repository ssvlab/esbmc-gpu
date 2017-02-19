/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with GNU Mailutils; if not, write to the Free
   Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

/*  Syntax:   timestamp [":before"/":after"] <header-name: string>
              <date: datestring>

    The "timestamp" test compares the value of a structured date header
    field with the given date.

    If the tagged argument is ":after" and the date from the header is
    after the specified date the result is true, otherwise, if the
    header date is before the given date, the result is false.

    If the tagged argument is ":before" and the date from the header is
    before the specified date the result is true, otherwise, if the
    header date is after the given date, the result is false.

    If no tagged argument is supplied, :after is assumed.

    Almost any date format is understood.

    Example:  timestamp :before "X-Expire-Timestamp" "now - 5 days"

    This test will return true, if the date in X-Expire-Timestamp is
    more than 5 days older than the current date. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  

#include <stdlib.h>
#include <mailutils/sieve.h>

/* Handler for the timestamp test */
static int
timestamp_test (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_sieve_value_t *h, *v;
  mu_header_t hdr;
  char *val;
  time_t now = time (NULL);
  time_t tlimit, tval;
  int rc;
  
  if (mu_sieve_get_debug_level (mach) & MU_SIEVE_DEBUG_TRACE)
    {
      mu_sieve_locus_t locus;
      mu_sieve_get_locus (mach, &locus);
      mu_sieve_debug (mach, "%s:%lu: TIMESTAMP\n",
		   locus.source_file,
		   (unsigned long) locus.source_line);
    }

  /* Retrieve required arguments: */
  /* First argument: header name */
  h = mu_sieve_value_get (args, 0);
  if (!h)
    {
      mu_sieve_arg_error (mach, 1);
      mu_sieve_abort (mach);
    }
  /* Second argument: date displacement */
  v = mu_sieve_value_get (args, 1);
  if (!v)
    {
      mu_sieve_arg_error (mach, 2);
      mu_sieve_abort (mach);
    }

  if (mu_parse_date (v->v.string, &tlimit, &now))
    {
      mu_sieve_error (mach, _("cannot parse date specification (%s)"),
		   v->v.string);
      mu_sieve_abort (mach);
    }

  rc = mu_message_get_header (mu_sieve_get_message (mach), &hdr);
  if (rc)
    {
      mu_sieve_error (mach, "mu_message_get_header: %s", mu_strerror (rc));
      mu_sieve_abort (mach);
    }
  
  if (mu_header_aget_value (hdr, h->v.string, &val))
    return 0;

  if (mu_parse_date (val, &tval, &now))
    {
      mu_sieve_error (mach,
		   "cannot parse header date specification (%s)",
		   val);
      free (val);
      mu_sieve_abort (mach);
    }
  free (val);

  rc = tval > tlimit;
    
  if (mu_sieve_tag_lookup (tags, "before", NULL))
    rc = !rc;  

  return rc;
}

/* Required arguments: */
static mu_sieve_data_type timestamp_req_args[] = {
  SVT_STRING,
  SVT_STRING,
  SVT_VOID
};

/* Tagged arguments: */
static mu_sieve_tag_def_t timestamp_tags[] = {
  { "after", SVT_VOID },
  { "before", SVT_VOID },
  { NULL }
};

static mu_sieve_tag_group_t timestamp_tag_groups[] = {
  { timestamp_tags, NULL },
  { NULL }
};

/* Initialization function. It is the only function exported from this
   module. */
int
SIEVE_EXPORT(timestamp,init) (mu_sieve_machine_t mach)
{
  return mu_sieve_register_test (mach, "timestamp", timestamp_test,
                              timestamp_req_args, timestamp_tag_groups, 1);
}
