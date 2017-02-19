/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifndef _MAILUTILS_LIBCFG_H
#define _MAILUTILS_LIBCFG_H

#include <mailutils/cfg.h>
#include <mailutils/gocs.h>
#include <mailutils/nls.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>

#ifndef MU_CONFIG_FILE
# define MU_CONFIG_FILE SYSCONFDIR "/mailutils.rc"
#endif

#ifndef MU_USER_CONFIG_FILE
# define MU_USER_CONFIG_FILE "~/.mailutils"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct mu_cfg_capa
{
  char *name;
  struct mu_cfg_param *cfgparam;
  mu_cfg_section_fp parser;
};

extern int mu_register_cfg_capa (const char *name,
				 struct mu_cfg_param *cfgparam,
				 mu_cfg_section_fp *parser);

extern void mu_libcfg_init (char **cnames);
extern int mu_parse_config_files (struct mu_cfg_param *param,
				  void *target_ptr) MU_CFG_DEPRECATED;
int mu_libcfg_parse_config (mu_cfg_tree_t **ptree);
  
extern void mu_acl_cfg_init (void);

#define __mu_common_cat2__(a,b) a ## b
#define __mu_common_cat3__(a,b,c) a ## b ## c
#define DCL_PARSER(capa)						      \
int									      \
__mu_common_cat3__(mu_,capa,_section_parser)				      \
     (enum mu_cfg_section_stage stage, const mu_cfg_node_t *node,	      \
      const char *section_label, void **section_data,                         \
      void *call_data, mu_cfg_tree_t *tree)	                              \
{									      \
  switch (stage)							      \
    {									      \
    case mu_cfg_section_start:						      \
      break;								      \
      									      \
    case mu_cfg_section_end:						      \
      mu_gocs_store (#capa, &__mu_common_cat2__(capa, _settings));	      \
    }									      \
  return 0;								      \
}

#define DCL_DEFAULT_CFG_CAPA(capa)                                            \
 struct mu_cfg_capa __mu_common_cat3__(mu_,capa,_cfg_capa) = {                \
      #capa,                                                                  \
      __mu_common_cat3__(mu_,capa,_param),                                    \
      __mu_common_cat3__(mu_,capa,_section_parser)                            \
 }

#define DCL_CFG_CAPA(capa)                                                    \
  DCL_PARSER (capa)                                                           \
  DCL_DEFAULT_CFG_CAPA (capa) 

extern struct mu_cfg_capa mu_mailbox_cfg_capa;
extern struct mu_cfg_capa mu_locking_cfg_capa;
extern struct mu_cfg_capa mu_address_cfg_capa;
extern struct mu_cfg_capa mu_mailer_cfg_capa;
extern struct mu_cfg_capa mu_logging_cfg_capa;
extern struct mu_cfg_capa mu_debug_cfg_capa;
extern struct mu_cfg_capa mu_gsasl_cfg_capa;
extern struct mu_cfg_capa mu_pam_cfg_capa;
extern struct mu_cfg_capa mu_radius_cfg_capa;
extern struct mu_cfg_capa mu_sql_cfg_capa;
extern struct mu_cfg_capa mu_tls_cfg_capa;
extern struct mu_cfg_capa mu_virtdomain_cfg_capa;
extern struct mu_cfg_capa mu_sieve_cfg_capa;
extern struct mu_cfg_capa mu_auth_cfg_capa;
extern struct mu_cfg_capa mu_ldap_cfg_capa;

#ifdef __cplusplus
}
#endif

#endif
