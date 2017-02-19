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
#include <mailutils/sql.h>
#include <mailutils/mutil.h>

static struct mu_sql_module_config sql_settings;



/* Resource file configuration */
static int
cb_password_type (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  
  if (mu_sql_decode_password_type (val->v.string, &sql_settings.password_type))
    mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			 _("unknown password type `%s'"),
			 val->v.string);
  return 0;
}

static int
_cb2_field_map (mu_debug_t debug, const char *arg, void *data)
{
  int err;
  int rc = mutil_parse_field_map (arg, &sql_settings.field_map, &err);
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

static struct mu_cfg_param mu_sql_param[] = {
  { "interface", mu_cfg_string, &sql_settings.interface, 0, NULL,
    N_("Set SQL interface to use (one of: mysql, odbc, or postgres).") },
  { "getpwnam", mu_cfg_string, &sql_settings.getpwnam_query, 0, NULL,
    N_("SQL query to use for getpwnam requests."),
    N_("query") },
  { "getpwuid", mu_cfg_string, &sql_settings.getpwuid_query, 0, NULL,
    N_("SQL query to use for getpwuid requests."),
    N_("query") },
  { "getpass", mu_cfg_string, &sql_settings.getpass_query, 0, NULL,
    N_("SQL query returning the user's password."),
    N_("query") },
  { "host", mu_cfg_string, &sql_settings.host, 0, NULL,
    N_("SQL server host name.") },
  { "user", mu_cfg_string, &sql_settings.user, 0, NULL,
    N_("SQL user name.") },
  { "passwd", mu_cfg_string, &sql_settings.passwd, 0, NULL,
    N_("Password for the SQL user.") },
  { "port", mu_cfg_int, &sql_settings.port, 0, NULL,
    N_("SQL server port.") },
  { "db", mu_cfg_string, &sql_settings.db, 0, NULL,
    N_("Database name.") },
  { "password-type", mu_cfg_callback, NULL, 0, cb_password_type,
    N_("Type of password returned by getpass query (one of: plain, hash, "
       "scrambled).") },
  { "positional", mu_cfg_bool, &sql_settings.positional, 0, NULL,
    N_("Use positional (v1.0 compatible) field interface.") },
  { "field-map", mu_cfg_callback, NULL, 0, cb_field_map,
    N_("Set a field-map for parsing SQL replies.  The map is a "
       "column-separated list of definitions.  Each definition has the "
       "following form:\n"
       "   <name: string>=<column: string>\n"
       "where <name> is one of the following: name, passwd, uid, gid, "
       "gecos, dir, shell, mailbox, quota, and <column> is the name of "
       "the corresponding SQL column."),
    N_("map") },
  { NULL }
};

DCL_CFG_CAPA (sql);

     
