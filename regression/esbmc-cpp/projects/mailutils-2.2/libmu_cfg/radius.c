/* This file is part of GNU Mailutils
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

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
#include <mailutils/radius.h>

static struct mu_radius_module_data radius_settings;

static struct mu_cfg_param mu_radius_param[] = {
  { "auth", mu_cfg_string, &radius_settings.auth_request, 0, NULL,
    N_("Radius request for authorization."),
    N_("request") },
  { "getpwnam", mu_cfg_string, &radius_settings.getpwnam_request, 0, NULL,
    N_("Radius request for getpwnam."),
    N_("request") },
  { "getpwuid", mu_cfg_string, &radius_settings.getpwuid_request, 0, NULL,
    N_("Radius request for getpwuid."),
    N_("request") },
  { "directory", mu_cfg_string, &radius_settings.config_dir, 0, NULL,
    N_("Set radius configuration directory.") },
  { NULL }
};

DCL_CFG_CAPA (radius);
