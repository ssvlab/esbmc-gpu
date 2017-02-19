/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2005, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

#ifndef _MAILUTILS_MU_AUTH_H
#define _MAILUTILS_MU_AUTH_H

#include <mailutils/types.h>
#include <mailutils/gocs.h>
#include <mailutils/debug.h>

#define MU_AUTH_NAME    "name"
#define MU_AUTH_PASSWD  "passwd"
#define MU_AUTH_UID     "uid"
#define MU_AUTH_GID     "gid"
#define MU_AUTH_GECOS   "gecos"
#define MU_AUTH_DIR     "dir"
#define MU_AUTH_SHELL   "shell"
#define MU_AUTH_MAILBOX "mailbox"
#define MU_AUTH_QUOTA   "quota"

struct mu_auth_data
{
  /* Where this info comes from: */
  const char *source;
  
  /* These are from struct passwd */
  char    *name;       /* user name */
  char    *passwd;     /* user password */
  uid_t   uid;         /* user id */
  gid_t   gid;         /* group id */
  char    *gecos;      /* real name */
  char    *dir;        /* home directory */
  char    *shell;      /* shell program */
  
  /* Additional fields */
  char    *mailbox;
  mu_off_t quota;

  int flags;
  
  int     change_uid;
};

#define MU_AF_QUOTA 0x1
#define MU_HAS_QUOTA(a) ((a)->flags & MU_AF_QUOTA)

typedef int (*mu_auth_fp) (struct mu_auth_data **data,
			   const void *key,
			   void *func_data,
			   void *call_data);

struct mu_auth_module
{
  char            *name;
  mu_gocs_init_fp init;
  mu_auth_fp      authenticate;
  void            *authenticate_data;
  mu_auth_fp      auth_by_name;
  void            *auth_by_name_data;
  mu_auth_fp      auth_by_uid;
  void            *auth_by_uid_data;
};

enum mu_auth_key_type
  {
    mu_auth_key_name,
    mu_auth_key_uid
  };

void mu_auth_begin_setup (void);
void mu_auth_finish_setup (void);

extern int mu_auth_runlist (mu_list_t flist,
			    struct mu_auth_data **return_data,
			    const void *key, void *data);

extern int mu_get_auth (struct mu_auth_data **auth, enum mu_auth_key_type type,
			const void *key);

extern struct mu_auth_data *
mu_get_auth_by_name (const char *username);

extern struct mu_auth_data *
mu_get_auth_by_uid (uid_t uid);

extern int
mu_authenticate (struct mu_auth_data *auth_data, const char *pass);

extern int mu_auth_nosupport (struct mu_auth_data **return_data,
			      const void *key,
			      void *func_data,
			      void *call_data);


extern void mu_auth_register_module (struct mu_auth_module *mod);

extern int mu_authorization_add_module (const char *modname);
extern void mu_authorization_add_module_list (const char *modlist);
extern int mu_authentication_add_module (const char *modname);
extern void mu_authentication_add_module_list (const char *modlist);
extern void mu_authentication_clear_list (void);
extern void mu_authorization_clear_list (void);

extern void mu_auth_init (void);
extern int mu_auth_data_alloc (struct mu_auth_data **ptr,
			       const char *name,
			       const char *passwd,
			       uid_t uid,
			       gid_t gid,
			       const char *gecos,
			       const char *dir,
		   	       const char *shell,
			       const char *mailbox,
			       int change_uid);
extern void mu_auth_data_set_quota (struct mu_auth_data *ptr, mu_off_t q);
extern void mu_auth_data_free (struct mu_auth_data *ptr);
extern void mu_auth_data_destroy (struct mu_auth_data **ptr);
extern mu_debug_t mu_auth_set_debug (mu_debug_t debug);


extern struct mu_auth_module mu_auth_system_module;
extern struct mu_auth_module mu_auth_generic_module;
extern struct mu_auth_module mu_auth_pam_module;
extern struct mu_auth_module mu_auth_sql_module;
extern struct mu_auth_module mu_auth_virtual_module;
extern struct mu_auth_module mu_auth_radius_module;
extern struct mu_auth_module mu_auth_ldap_module;

#define MU_AUTH_REGISTER_ALL_MODULES() do {\
  mu_auth_register_module (&mu_auth_generic_module); \
  mu_auth_register_module (&mu_auth_system_module); \
  mu_auth_register_module (&mu_auth_pam_module);\
  mu_auth_register_module (&mu_auth_sql_module);\
  mu_auth_register_module (&mu_auth_virtual_module);\
  mu_auth_register_module (&mu_auth_radius_module);\
  mu_auth_register_module (&mu_auth_ldap_module);\
  } while (0)

#endif
