/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <syslog.h>
#include <string.h>
#include <mailutils/debug.h>
#include <mailutils/nls.h>
#include <mailutils/syslog.h>
#include <mailutils/cfg.h>
#include <mailutils/diag.h>
#include <mailutils/error.h>

int mu_tcp_wrapper_enable = 1;
char *mu_tcp_wrapper_daemon;

#ifdef WITH_LIBWRAP
# include <tcpd.h>
int deny_severity = LOG_INFO;
int allow_severity = LOG_INFO;

int
mu_tcp_wrapper_cb_hosts_allow_syslog (mu_debug_t debug, void *data,
				      mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (mu_string_to_syslog_priority (val->v.string, &allow_severity))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
			 _("unknown syslog priority `%s'"), 
			 val->v.string);
  return 0;
}

int
mu_tcp_wrapper_cb_hosts_deny_syslog (mu_debug_t debug, void *data,
				     mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (mu_string_to_syslog_priority (val->v.string, &deny_severity))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, 
			 _("unknown syslog priority `%s'"), 
			 val->v.string);
  return 0;
}

int
mu_tcpwrapper_access (int fd)
{
  struct request_info req;

  if (!mu_tcp_wrapper_enable)
    return 1;
  request_init (&req,
		RQ_DAEMON,
		mu_tcp_wrapper_daemon ?
		     mu_tcp_wrapper_daemon : mu_program_name,
		RQ_FILE, fd, NULL);
  fromhost (&req);
  return hosts_access (&req);
}

struct mu_cfg_param tcpwrapper_param[] = {
  { "enable", mu_cfg_bool, &mu_tcp_wrapper_enable, 0, NULL,	      
    N_("Enable TCP wrapper access control.  Default is \"yes\".") },	      
  { "daemon", mu_cfg_string, &mu_tcp_wrapper_daemon, 0, NULL,     
    N_("Set daemon name for TCP wrapper lookups.  Default is program name."), 
    N_("name") },							      
  { "allow-table", mu_cfg_string, &hosts_allow_table,
    0, NULL,
    N_("Use file for positive client address access control "		      
       "(default: /etc/hosts.allow)."),					      
    N_("file") },							      
  { "deny-table", mu_cfg_string, &hosts_deny_table,
    0, NULL,                                             
    N_("Use file for negative client address access control "		      
       "(default: /etc/hosts.deny)."),					      
    N_("file") },							      
  { "allow-syslog-priority", mu_cfg_callback, NULL, 0,	       	      
    mu_tcp_wrapper_cb_hosts_allow_syslog,				      
    N_("Log host allows at this syslog priority."),
    N_("level") },							      
  { "deny-syslog-priority", mu_cfg_callback, NULL, 0,			      
    mu_tcp_wrapper_cb_hosts_deny_syslog,				      
    N_("Log host denies at this syslog priority."),
    N_("level") },
  { NULL }
};

void
mu_tcpwrapper_cfg_init ()
{
  struct mu_cfg_section *section;
  mu_create_canned_section ("tcp-wrappers", &section);
  mu_cfg_section_add_params (section, tcpwrapper_param);
}

#else

void
mu_tcpwrapper_cfg_init ()
{
}

int
mu_tcpwrapper_access (int fd)
{
  return 1;
}

#endif

int
mu_tcp_wrapper_prefork (int fd, void *data, struct sockaddr *sa, int salen)
{
  if (mu_tcp_wrapper_enable
      && sa->sa_family == AF_INET
      && !mu_tcpwrapper_access (fd))
    {
      char *p = mu_sockaddr_to_astr (sa, salen);
      mu_error (_("access from %s blocked by TCP wrappers"), p);
      free (p);
      return 1;
    }
  return 0;
}
     
