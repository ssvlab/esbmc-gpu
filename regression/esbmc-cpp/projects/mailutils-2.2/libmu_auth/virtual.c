/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2006, 2007, 2008, 2010 Free Software Foundation,
   Inc.

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
#ifdef HAVE_SECURITY_PAM_APPL_H
# include <security/pam_appl.h>
#endif
#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h> /*FIXME!*/

#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <mailutils/mailbox.h>
#include <mailutils/mu_auth.h>
#include <mailutils/nls.h>
#include <mailutils/errno.h>

#ifdef ENABLE_VIRTUAL_DOMAINS

struct mu_gocs_virtual mu_virtual_module_config = { SITE_VIRTUAL_PWDDIR };

int
mu_virtual_module_init (enum mu_gocs_op op, void *data)
{
  if (op == mu_gocs_op_set && data)
    {
      struct mu_gocs_virtual *p = data;
      mu_virtual_module_config = *p;
    }
  return 0;
}

#if !HAVE_FGETPWENT
/* FIXME: A temporary solution. Need proper declaration in .h */
extern struct passwd *mu_fgetpwent (FILE *fp);
#define fgetpwent mu_fgetpwent
#endif

static struct passwd *
getpwnam_virtual (const char *u)
{
  struct passwd *pw = NULL;
  FILE *pfile;
  size_t i = 0, len = strlen (u), delim = 0;
  char *filename;

  for (i = 0; i < len && delim == 0; i++)
    if (u[i] == '!' || u[i] == ':' || u[i] == '@')
      delim = i;

  if (delim == 0)
    return NULL;

  filename = malloc (strlen (mu_virtual_module_config.pwddir) +
		     strlen (&u[delim + 1]) + 2 /* slash and null byte */);
  if (filename == NULL)
    return NULL;

  sprintf (filename, "%s/%s", mu_virtual_module_config.pwddir, &u[delim + 1]);
  pfile = fopen (filename, "r");
  free (filename);

  if (pfile)
    {
      while ((pw = fgetpwent (pfile)) != NULL)
	{
	  if (strlen (pw->pw_name) == delim
	      && strncmp (u, pw->pw_name, delim) == 0)
	    break;
	}
      fclose (pfile);
    }
  return pw;
}

static struct passwd *
getpwnam_ip_virtual (const char *u)
{
  struct sockaddr_in addr;
  struct passwd *pw = NULL;
  socklen_t len = sizeof (addr);
  
  if (getsockname (0, (struct sockaddr *)&addr, &len) == 0)
    {
      char *ip;
      char *user;

      struct hostent *info = gethostbyaddr ((char *)&addr.sin_addr,
					    4, AF_INET);

      if (info)
	{
	  user = malloc (strlen (info->h_name) + strlen (u) + 2);
	  if (user)
	    {
	      sprintf (user, "%s!%s", u, info->h_name);
	      pw = getpwnam_virtual (user);
	      free (user);
	    }
        }

      if (!pw)
	{
	  ip = inet_ntoa (addr.sin_addr);
	  user = malloc (strlen (ip) + strlen (u) + 2);
	  if (user)
	    {
	      sprintf (user, "%s!%s", u, ip);
	      pw = getpwnam_virtual (user);
	      free (user);
	    }
	}
    }
  return pw;
}

/* Virtual domains */
static int
mu_auth_virt_domain_by_name (struct mu_auth_data **return_data,
			     const void *key,
			     void *unused_func_data, void *unused_call_data)
{
  int rc;
  struct passwd *pw;
  char *mailbox_name;
  
  if (!key)
    return EINVAL;

  pw = getpwnam_virtual (key);
  if (!pw)
    {
      pw = getpwnam_ip_virtual (key);
      if (!pw)
	return MU_ERR_AUTH_FAILURE;
    }
  
  mailbox_name = calloc (strlen (pw->pw_dir) + strlen ("/INBOX") + 1, 1);
  if (!mailbox_name)
    return ENOMEM;
  sprintf (mailbox_name, "%s/INBOX", pw->pw_dir);

  rc = mu_auth_data_alloc (return_data,
			   pw->pw_name,
			   pw->pw_passwd,
			   pw->pw_uid,
			   pw->pw_gid,
			   pw->pw_gecos,
			   pw->pw_dir,
			   pw->pw_shell,
			   mailbox_name,
			   0);
  free (mailbox_name);
  return rc;
}

#else
static int
mu_auth_virt_domain_by_name (struct mu_auth_data **return_data MU_ARG_UNUSED,
			     const void *key MU_ARG_UNUSED,
			     void *func_data MU_ARG_UNUSED,
			     void *call_data MU_ARG_UNUSED)
{
  return ENOSYS;
}
#endif

struct mu_auth_module mu_auth_virtual_module = {
  "virtdomain",
#ifdef ENABLE_VIRTUAL_DOMAINS
  mu_virtual_module_init,
#else
  NULL,
#endif
  mu_auth_nosupport,
  NULL,
  mu_auth_virt_domain_by_name,
  NULL,
  mu_auth_nosupport,
  NULL
};
    
