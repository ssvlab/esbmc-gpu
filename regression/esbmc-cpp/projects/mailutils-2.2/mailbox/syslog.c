/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3 of the License, or (at your
   option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program. If not, see <http://www.gnu.org/licenses/>. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <syslog.h>
#include <string.h>
#include <mailutils/diag.h>
#include <mailutils/kwd.h>
#include <mailutils/syslog.h>
#include <mailutils/cstr.h>

#ifndef LOG_AUTHPRIV
# define LOG_AUTHPRIV
#endif

static mu_kwd_t kw_facility[] = {
  { "USER",    LOG_USER },   
  { "DAEMON",  LOG_DAEMON },
  { "AUTH",    LOG_AUTH },
  { "AUTHPRIV",LOG_AUTHPRIV },
  { "MAIL",    LOG_MAIL },
  { "CRON",    LOG_CRON },
  { "LOCAL0",  LOG_LOCAL0 },
  { "LOCAL1",  LOG_LOCAL1 },
  { "LOCAL2",  LOG_LOCAL2 },
  { "LOCAL3",  LOG_LOCAL3 },
  { "LOCAL4",  LOG_LOCAL4 },
  { "LOCAL5",  LOG_LOCAL5 },
  { "LOCAL6",  LOG_LOCAL6 },
  { "LOCAL7",  LOG_LOCAL7 },
  { NULL }
};

static int
syslog_to_n (mu_kwd_t *kw, const char *str, int *pint)
{
  if (mu_c_strncasecmp (str, "LOG_", 4) == 0)
    str += 4;
  return mu_kwd_xlat_name_ci (kw, str, pint);
}

int
mu_string_to_syslog_facility (const char *str, int *pfacility)
{
  return syslog_to_n (kw_facility, str, pfacility);
}

const char *
mu_syslog_facility_to_string (int n)
{
  const char *res = NULL;
  mu_kwd_xlat_tok (kw_facility, n, &res);
  return res;
}

static mu_kwd_t kw_prio[] = {
  { "EMERG", LOG_EMERG },
  { "ALERT", LOG_ALERT },
  { "CRIT", LOG_CRIT },
  { "ERR", LOG_ERR },
  { "WARNING", LOG_WARNING },
  { "NOTICE", LOG_NOTICE },
  { "INFO", LOG_INFO },
  { "DEBUG", LOG_DEBUG },
  { NULL }
};

int
mu_string_to_syslog_priority (const char *str, int *pprio)
{
  return syslog_to_n (kw_prio, str, pprio);
}

const char *
mu_syslog_priority_to_string (int n)
{
  const char *res = NULL;
  mu_kwd_xlat_tok (kw_prio, n, &res);
  return res;
}

int
mu_diag_level_to_syslog (mu_log_level_t level)
{
  switch (level)
    {
    case MU_DIAG_EMERG:
      return LOG_EMERG;
      
    case MU_DIAG_ALERT:
      return LOG_ALERT;
	
    case MU_DIAG_CRIT:
      return LOG_CRIT;
      
    case MU_DIAG_ERROR:
      return LOG_ERR;
      
    case MU_DIAG_WARNING:
      return LOG_WARNING;
      
    case MU_DIAG_NOTICE:
      return LOG_NOTICE;
      
    case MU_DIAG_INFO:
      return LOG_INFO;
      
    case MU_DIAG_DEBUG:
      return LOG_DEBUG;
    }
  return LOG_EMERG;
}

int
mu_diag_syslog_printer (void *data, mu_log_level_t level, const char *buf)
{
  int len = strlen (buf);
  if (len > 0 && buf[len-1] == '\n')
    {
      len--;
      if (len > 0 && buf[len-1] == '\r')
	len--;
    }
  syslog (mu_diag_level_to_syslog (level), "%-.*s", len, buf);
  return 0;
}


int mu_log_facility = LOG_FACILITY;
char *mu_log_tag = NULL;
