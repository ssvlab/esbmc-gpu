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

static struct mu_gocs_virtual virtdomain_settings;

static struct mu_cfg_param mu_virtdomain_param[] = {
  { "passwd-dir", mu_cfg_string, &virtdomain_settings, 0, NULL,
    N_("Name of the directory where virtual domain password files are "
       "located."),
    N_("dir") },
  { NULL }
};

DCL_CFG_CAPA (virtdomain);
     
