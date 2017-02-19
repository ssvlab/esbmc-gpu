/* This file is part of GNU Mailutils
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <string.h>
#include "mailutils/libcfg.h"
#include <mailutils/debug.h>
#include <mailutils/syslog.h>
#include <mailutils/mailbox.h>
#include <mailutils/io.h>

static struct mu_gocs_locking locking_settings;
static struct mu_gocs_logging logging_settings;
static struct mu_gocs_mailbox mailbox_settings;
static struct mu_gocs_source_email address_settings;
static struct mu_gocs_mailer mailer_settings;
static struct mu_gocs_debug debug_settings;


/* ************************************************************************* */
/* Mailbox                                                                   */
/* ************************************************************************* */

static int
_cb_folder (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  mu_set_folder_directory (val->v.string);
  return 0;
}

static struct mu_cfg_param mu_mailbox_param[] = {
  { "mail-spool", mu_cfg_string, &mailbox_settings.mail_spool, 0, NULL,
    N_("Use specified URL as a mailspool directory."),
    N_("url") },
  { "mailbox-pattern", mu_cfg_string, &mailbox_settings.mailbox_pattern,
    0, NULL,
    N_("Create mailbox URL using <pattern>."),
    N_("pattern") },
  { "mailbox-type", mu_cfg_string, &mailbox_settings.mailbox_type, 0, NULL,
    N_("Default mailbox type."), N_("protocol") },
  { "folder", mu_cfg_callback, NULL, 0, _cb_folder,
    N_("Default user mail folder"),
    N_("dir") },
  { NULL }
};

DCL_CFG_CAPA (mailbox);


/* ************************************************************************* */
/* Locking                                                                   */
/* ************************************************************************* */

static struct mu_cfg_param mu_locking_param[] = {
  /* FIXME: Flags are superfluous. */
  { "flags", mu_cfg_string, &locking_settings.lock_flags, 0, NULL,
    N_("Default locker flags (E=external, R=retry, T=time, P=pid).") },
  { "retry-timeout", mu_cfg_ulong, &locking_settings.lock_retry_timeout,
    0, NULL,
    N_("Set timeout for acquiring the lock.") },
  { "retry-count", mu_cfg_ulong, &locking_settings.lock_retry_count, 0, NULL,
    N_("Set the maximum number of times to retry acquiring the lock.") },
  { "expire-timeout", mu_cfg_ulong, &locking_settings.lock_expire_timeout,
    0, NULL,
    N_("Expire locks older than this amount of time.") },
  { "external-locker", mu_cfg_string, &locking_settings.external_locker, 
    0, NULL,
    N_("Use external locker program."),
    N_("prog") },
  { NULL, }
};

DCL_CFG_CAPA (locking);


/* ************************************************************************* */
/* Address                                                                   */
/* ************************************************************************* */
     
static struct mu_cfg_param mu_address_param[] = {
  { "email-addr", mu_cfg_string, &address_settings.address, 0, NULL,
    N_("Set the current user email address (default is "
       "loginname@defaultdomain)."),
    N_("email") },
  { "email-domain", mu_cfg_string, &address_settings.domain, 0, NULL,
    N_("Set e-mail domain for unqualified user names (default is this host)"),
    N_("domain") },
  { NULL }
};

DCL_CFG_CAPA (address);

     
/* ************************************************************************* */
/* Mailer                                                                    */
/* ************************************************************************* */
     
static struct mu_cfg_param mu_mailer_param[] = {
  { "url", mu_cfg_string, &mailer_settings.mailer, 0, NULL,
    N_("Use this URL as the default mailer"),
    N_("url") },
  { NULL }
};

DCL_CFG_CAPA (mailer);


/* ************************************************************************* */
/* Logging                                                                   */
/* ************************************************************************* */

int
cb_facility (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  
  if (mu_string_to_syslog_facility (val->v.string, &logging_settings.facility))
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
                           _("unknown syslog facility `%s'"), 
			   val->v.string);
      return 1;
    }
   return 0;
}

static struct mu_cfg_param mu_logging_param[] = {
  { "facility", mu_cfg_callback, NULL, 0, cb_facility,
    N_("Set syslog facility. Arg is one of the following: user, daemon, "
       "auth, authpriv, mail, cron, local0 through local7 (case-insensitive), "
       "or a facility number.") },
  { "tag", mu_cfg_string, &mu_log_tag, 0, NULL,
    N_("Tag syslog messages with this string.") },
  { NULL }
};

DCL_CFG_CAPA (logging);


/* ************************************************************************* */
/* Debug                                                                     */
/* ************************************************************************* */

static int
_cb2_debug_level (mu_debug_t debug, const char *arg, void *data MU_ARG_UNUSED)
{
  char *pfx;
  struct mu_debug_locus locus;

  if (debug_settings.string)
    free (debug_settings.string);
  debug_settings.string = strdup (arg);
  if (mu_debug_get_locus (debug, &locus) == 0)
    {
      int status = mu_asprintf (&pfx, "%s:%lu",
				locus.file, (unsigned long) locus.line);
      if (status)
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       "%s", mu_strerror (status));
	  return 1;
	}
    }
  else
    pfx = strdup ("command line");/*FIXME*/
  /*FIXME: this is suboptimal, there's no use parsing 1st arg in
    mu_global_debug_from_string */
  mu_global_debug_from_string (debug_settings.string, pfx);
  free (pfx);
  free (debug_settings.string);
  free (debug_settings.errpfx);
  memset (&debug_settings, 0, sizeof debug_settings);
  return 0;
}

static int
cb_debug_level (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  return mu_cfg_string_value_cb (debug, val, _cb2_debug_level, NULL);
}

static struct mu_cfg_param mu_debug_param[] = {
  { "level", mu_cfg_callback, NULL, 0, &cb_debug_level,
    N_("Set Mailutils debugging level.  Argument is a colon-separated list "
       "of debugging specifications in the form:\n"
       "   <object: string>[[:]=<level: number>].") },
  { "line-info", mu_cfg_bool, &debug_settings.line_info, 0, NULL,
    N_("Prefix debug messages with Mailutils source locations.") },
  { NULL }
};

DCL_CFG_CAPA (debug);
