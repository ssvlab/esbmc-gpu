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
#include <mailutils/gsasl.h>

static struct mu_gsasl_module_data gsasl_settings;

static struct mu_cfg_param mu_gsasl_param[] = {
  { "cram-passwd", mu_cfg_string, &gsasl_settings.cram_md5_pwd, 0, NULL,
    N_("Name of GSASL password file."),
    N_("file") },
  { "service", mu_cfg_string, &gsasl_settings.service, 0, NULL,
    N_("SASL service name."),
    N_("name") },
  { "realm", mu_cfg_string, &gsasl_settings.realm, 0, NULL,
    N_("SASL realm name."),
    N_("name") },
  { "hostname", mu_cfg_string, &gsasl_settings.hostname, 0, NULL,
    N_("SASL host name."),
    N_("name") },
  { "anonymous-user", mu_cfg_string, &gsasl_settings.anon_user, 0, NULL,
    N_("Anonymous user name."),
    N_("name") },

  { NULL }
};

DCL_CFG_CAPA (gsasl);
     
     
