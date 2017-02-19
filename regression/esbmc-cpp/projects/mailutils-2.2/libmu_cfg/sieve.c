/* This file is part of GNU Mailutils
   Copyright (C) 2007, 2008, 2010 Free Software Foundation, Inc.

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
#include <mailutils/sieve.h>

static struct mu_gocs_sieve sieve_settings;

static int
cb_clear_library_path (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  int flag;

  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (mu_cfg_parse_boolean (val->v.string, &flag))
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("not a boolean"));
      return 1;
    }
  if (flag)
    sieve_settings.clearflags |= MU_SIEVE_CLEAR_LIBRARY_PATH;
  return 0;
}

static int
cb_clear_include_path (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  int flag;
  
  if (mu_cfg_assert_value_type (val, MU_CFG_STRING, debug))
    return 1;
  if (mu_cfg_parse_boolean (val->v.string, &flag))
    {
      mu_cfg_format_error (debug, MU_DEBUG_ERROR, _("not a boolean"));
      return 1;
    }
  if (flag)
    sieve_settings.clearflags |= MU_SIEVE_CLEAR_INCLUDE_PATH;
  return 0;
}

static int
_add_path (mu_debug_t debug, const char *arg, void *data)
{
  char *p, *tmp;
  mu_list_t *plist = data;
    
  if (!*plist)
    {
      int rc = mu_list_create (plist);
      if (rc)
	{
	  mu_cfg_format_error (debug, MU_DEBUG_ERROR,
			       _("cannot create list: %s"), mu_strerror (rc));
	  exit (1);
	}
      mu_list_set_destroy_item (*plist, mu_list_free_item);
    }
  /* FIXME: Use mu_argcv */
  tmp = strdup (arg);
  for (p = strtok (tmp, ":"); p; p = strtok (NULL, ":"))
    mu_list_append (*plist, strdup (p));
  free (tmp);
  return 0;
}

static int
cb_include_path (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  return mu_cfg_string_value_cb (debug, val, _add_path,
				 &sieve_settings.include_path);
}  

static int
cb_library_path (mu_debug_t debug, void *data, mu_config_value_t *val)
{
  return mu_cfg_string_value_cb (debug, val, _add_path,
				 &sieve_settings.library_path);
}

static struct mu_cfg_param mu_sieve_param[] = {
  { "clear-library-path", mu_cfg_callback, NULL, 0, cb_clear_library_path,
     N_("Clear library search path.") },
  { "clear-include-path", mu_cfg_callback, NULL, 0, cb_clear_include_path,
     N_("Clear include search path.") },
  { "library-path", mu_cfg_callback, NULL, 0, cb_library_path,
    N_("Add directories to the library search path.  Argument is a "
       "colon-separated list of directories."),
    N_("list") },
  { "include-path", mu_cfg_callback, NULL, 0, cb_include_path,
    N_("Add directories to the include search path.  Argument is a "
       "colon-separated list of directories."),
    N_("list") },
  { NULL }
};

DCL_CFG_CAPA (sieve);
