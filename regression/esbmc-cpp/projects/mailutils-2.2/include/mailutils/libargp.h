/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2008, 2010 Free
   Software Foundation, Inc.

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

#ifndef _MAILUTILS_LIBARGP_H
#define _MAILUTILS_LIBARGP_H

#include "mailutils/types.h"
#include "mailutils/gocs.h"
#include "mailutils/nls.h"
#include "mailutils/error.h"
#include "mailutils/errno.h"
#include "mailutils/version.h"
#include "argp.h"
#include "errno.h" 
#include "strings.h"

#include "mailutils/libcfg.h"

#ifdef __cplusplus
extern "C" {
#endif

struct mu_cmdline_capa
{
  char *name;
  struct argp_child *child;
};

extern int mu_help_config_mode;
extern int mu_rcfile_lint;
extern int (*mu_app_cfg_verifier) (void);
  
extern struct mu_cmdline_capa mu_common_cmdline;
extern struct mu_cmdline_capa mu_logging_cmdline;
extern struct mu_cmdline_capa mu_license_cmdline;
extern struct mu_cmdline_capa mu_mailbox_cmdline;
extern struct mu_cmdline_capa mu_locking_cmdline;
extern struct mu_cmdline_capa mu_address_cmdline;
extern struct mu_cmdline_capa mu_mailer_cmdline;
extern struct mu_cmdline_capa mu_sieve_cmdline;
extern struct mu_cmdline_capa mu_debug_cmdline;
  
extern struct mu_cmdline_capa mu_pam_cmdline;
extern struct mu_cmdline_capa mu_gsasl_cmdline;
extern struct mu_cmdline_capa mu_tls_cmdline;
extern struct mu_cmdline_capa mu_radius_cmdline;
extern struct mu_cmdline_capa mu_sql_cmdline;
extern struct mu_cmdline_capa mu_virtdomain_cmdline;
extern struct mu_cmdline_capa mu_auth_cmdline;

extern void mu_libargp_init (void);
  
extern struct argp *mu_argp_build (const struct argp *argp, char ***pcapa);
extern void mu_argp_done (struct argp *argp);
  
extern int mu_register_argp_capa (const char *name, struct argp_child *child);

void mu_argp_init (const char *vers, const char *bugaddr);
int mu_app_init (struct argp *myargp, const char **capa,
		 struct mu_cfg_param *cfg_param, 
		 int argc, char **argv, int flags, int *pindex, void *data);

error_t mu_argp_parse (const struct argp *myargp, 
		       int *pargc, char **pargv[],  
		       unsigned flags,
		       const char *capa[],
		       int *arg_index,     
		       void *input) __attribute__ ((deprecated));

void mu_argp_node_list_init (mu_list_t *);
void mu_argp_node_list_add (mu_list_t, mu_cfg_node_t *);
void mu_argp_node_list_new (mu_list_t, const char *, const char *);
void mu_argp_node_list_finish (mu_list_t, char *, char *);
  
#ifdef __cplusplus
}
#endif

#endif

