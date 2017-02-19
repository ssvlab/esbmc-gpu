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
#include <mailutils/tls.h>

static struct mu_tls_module_config tls_settings;

static struct mu_cfg_param mu_tls_param[] = {
  { "enable", mu_cfg_bool, &tls_settings.enable, 0, NULL,
    N_("Enable client TLS encryption.") },
  { "ssl-cert", mu_cfg_string, &tls_settings.ssl_cert, 0, NULL,
    N_("Specify SSL certificate file."),
    N_("file") },
  { "ssl-key", mu_cfg_string, &tls_settings.ssl_key, 0, NULL,
    N_("Specify SSL certificate key file."),
    N_("file") },
  { "ssl-cafile", mu_cfg_string, &tls_settings.ssl_cafile, 0, NULL,
    N_("Specify trusted CAs file."),
    N_("file") },
  { NULL }
}; 

DCL_CFG_CAPA (tls);

