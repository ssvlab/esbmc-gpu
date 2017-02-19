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
#include "mailutils/libcfg.h"
#include "mailutils/mutil.h"
#include "mailutils/ldap.h"

static struct mu_ldap_module_config ldap_settings;

static int
_cb2_field_map (mu_debug_t debug, const char *arg, void *data)
{
  int err;
  int rc = mutil_parse_field_map (arg, &ldap_settings.field_map, &err);
  if (rc)
    /* FIXME: this message may be misleading */
    mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("error near element %d: %s"),
			 err, mu_strerror (rc));
  return 0;
}

static int
cb_field_map (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  return mu_cfg_string_value_cb (debug, val, _cb2_field_map, NULL);
}

static struct mu_cfg_param mu_ldap_param[] = {
  { "enable", mu_cfg_bool, &ldap_settings.enable, 0, NULL,
    N_("Enable LDAP lookups.") },
  { "url", mu_cfg_string, &ldap_settings.url, 0, NULL,
    N_("Set URL of the LDAP server."),
    N_("url") },
  { "base", mu_cfg_string, &ldap_settings.base, 0, NULL,
    N_("Base DN for LDAP lookups."),
    N_("dn") },
  { "binddn", mu_cfg_string, &ldap_settings.binddn, 0, NULL,
    N_("DN for accessing LDAP database."),
    N_("dn") },
  { "passwd", mu_cfg_string, &ldap_settings.passwd, 0, NULL,
    N_("Password for use with binddn.") },
  { "tls", mu_cfg_bool, &ldap_settings.tls, 0, NULL,
    N_("Use TLS encryption.") },
  { "debug", mu_cfg_int, &ldap_settings.debug, 0, NULL,
    N_("Set LDAP debugging level.") },
  { "field-map", mu_cfg_callback, NULL, 0, cb_field_map,
    N_("Set a field-map for parsing LDAP replies.  The map is a "
       "column-separated list of definitions.  Each definition has the "
       "following form:\n"
       "   <name: string>=<attr: string>\n"
       "where <name> is one of the following: name, passwd, uid, gid, "
       "gecos, dir, shell, mailbox, quota, and <attr> is the name of "
       "the corresponding LDAP attribute."),
    N_("map") },
  { "getpwnam", mu_cfg_string, &ldap_settings.getpwnam_filter, 0, NULL,
    N_("LDAP filter to use for getpwnam requests."),
    N_("filter") },
  { "getpwuid", mu_cfg_string, &ldap_settings.getpwuid_filter, 0, NULL,
    N_("LDAP filter to use for getpwuid requests."),
    N_("filter") },
  { NULL }
};

int									      
mu_ldap_section_parser
   (enum mu_cfg_section_stage stage, const mu_cfg_node_t *node,	      
    const char *section_label, void **section_data,
    void *call_data, mu_cfg_tree_t *tree)
{									      
  switch (stage)							      
    {									      
    case mu_cfg_section_start:
      ldap_settings.enable = 1;
      break;								      
      									      
    case mu_cfg_section_end:						      
      mu_gocs_store ("ldap", &ldap_settings);	      
    }									      
  return 0;								      
}

struct mu_cfg_capa mu_ldap_cfg_capa = {                
  "ldap",  mu_ldap_param, mu_ldap_section_parser
};
