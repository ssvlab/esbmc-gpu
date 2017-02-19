/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2007, 2009, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#ifdef HAVE_SHADOW_H
# include <shadow.h>
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif

#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <mailutils/mailbox.h>
#include <mailutils/mu_auth.h>
#include <mailutils/errno.h>

/* System database */
static int
mu_auth_system (struct mu_auth_data **return_data, const struct passwd *pw)
{
  char *mailbox_name;
  int rc;
  
  if (!pw)
    return MU_ERR_AUTH_FAILURE;

  rc = mu_construct_user_mailbox_url (&mailbox_name, pw->pw_name);
  if (rc)
    return rc; /* FIXME: Return code is lost */
  
  rc = mu_auth_data_alloc (return_data,
			   pw->pw_name,
			   pw->pw_passwd,
			   pw->pw_uid,
			   pw->pw_gid,
			   pw->pw_gecos,
			   pw->pw_dir,
			   pw->pw_shell,
			   mailbox_name,
			   1);
  free (mailbox_name);
  return rc;
}

int
mu_auth_system_by_name (struct mu_auth_data **return_data,
			const void *key,
			void *func_data MU_ARG_UNUSED,
			void *call_data MU_ARG_UNUSED)
{
  if (!key)
    return EINVAL;
  return mu_auth_system (return_data, getpwnam (key));
}

static int
mu_auth_system_by_uid (struct mu_auth_data **return_data,
		       const void *key,
		       void *func_data MU_ARG_UNUSED,
		       void *call_data MU_ARG_UNUSED)
{
  if (!key)
    return EINVAL;
  return mu_auth_system (return_data, getpwuid (*(uid_t*) key));
}

static int
mu_authenticate_generic (struct mu_auth_data **return_data MU_ARG_UNUSED,
			 const void *key,
			 void *func_data MU_ARG_UNUSED,
			 void *call_data)
{
  const struct mu_auth_data *auth_data = key;
  char *pass = call_data;

  if (!auth_data || !pass)
    return EINVAL;
  
  return auth_data->passwd
         && strcmp (auth_data->passwd, crypt (pass, auth_data->passwd)) == 0 ?
          0 : MU_ERR_AUTH_FAILURE;
}

/* Called only if generic fails */
static int
mu_authenticate_system (struct mu_auth_data **return_data MU_ARG_UNUSED,
			const void *key,
			void *func_data MU_ARG_UNUSED,
			void *call_data)
{
  const struct mu_auth_data *auth_data = key;

#ifdef HAVE_SHADOW_H
  char *pass = call_data;
  
  if (auth_data)
    {
      struct spwd *spw;
      spw = getspnam (auth_data->name);
      if (spw)
	return strcmp (spw->sp_pwdp, crypt (pass, spw->sp_pwdp)) == 0 ?
	        0 : MU_ERR_AUTH_FAILURE;
    }
#endif
  return MU_ERR_AUTH_FAILURE;
}


struct mu_auth_module mu_auth_system_module = {
  "system",
  NULL,
  mu_authenticate_system,
  NULL,
  mu_auth_system_by_name,
  NULL,
  mu_auth_system_by_uid,
  NULL
};


struct mu_auth_module mu_auth_generic_module = {
  "generic",
  NULL,
  mu_authenticate_generic,
  NULL,
  mu_auth_nosupport,
  NULL,
  mu_auth_nosupport,
  NULL
};

