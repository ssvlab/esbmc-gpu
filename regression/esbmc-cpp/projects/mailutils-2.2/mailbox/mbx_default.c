/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <pwd.h>
#include <unistd.h>

#include <confpaths.h>

#include <mailutils/mailbox.h>
#include <mailutils/mutil.h>
#include <mailutils/debug.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/mu_auth.h>
#include <mailutils/vartab.h>
#include <mailutils/folder.h>
#include <mailutils/auth.h>

#include <mailbox0.h>

char *mu_ticket_file = "~/.mu-tickets";

static char *_mu_mailbox_pattern;

static char *_default_folder_dir = "Mail";
static char *_mu_folder_dir;

static int
mu_normalize_mailbox_url (char **pout, const char *dir)
{
  int len;
  int addslash = 0;
#define USERSUFFIX "${user}"
  
  if (!pout)
    return MU_ERR_OUT_PTR_NULL;
      
  len = strlen (dir);
  if (dir[len-1] == '=')
    {
      if (len > 5 && strcmp (dir + len - 5, "user=") == 0)
	*pout = strdup (dir);
      else
	return MU_ERR_BAD_FILENAME;
    }
  else if (dir[len-1] != '/')
    addslash = 1;

  *pout = malloc (strlen (dir) + (addslash ? 1 : 0) + sizeof USERSUFFIX);
  if (!*pout)
    return ENOMEM;

  strcpy (*pout, dir);
  if (addslash)
    strcat (*pout, "/");
  strcat (*pout, USERSUFFIX);
#undef USERSUFFIX
  return 0;
}

int
mu_set_mail_directory (const char *p)
{
  if (_mu_mailbox_pattern)
    free (_mu_mailbox_pattern);
  if (!p)
    {
      _mu_mailbox_pattern = NULL;
      return 0;
    }
  return mu_normalize_mailbox_url (&_mu_mailbox_pattern, p);
}

int
mu_set_mailbox_pattern (const char *pat)
{
  if (_mu_mailbox_pattern)
    free (_mu_mailbox_pattern);
  if (!pat)
    {
      _mu_mailbox_pattern = NULL;
      return 0;
    }
  _mu_mailbox_pattern = strdup (pat);
  return _mu_mailbox_pattern ? 0 : ENOMEM;
}

void
mu_set_folder_directory (const char *p)
{
  if (_mu_folder_dir != _default_folder_dir)
    free (_mu_folder_dir);
  _mu_folder_dir = strdup (p);
}

const char *
mu_mailbox_url ()
{
  if (!_mu_mailbox_pattern)
    mu_set_mail_directory (MU_PATH_MAILDIR);
  return _mu_mailbox_pattern;
}

const char *
mu_folder_directory ()
{
  if (!_mu_folder_dir)
    _mu_folder_dir = _default_folder_dir;
  return _mu_folder_dir;
}

int
mu_construct_user_mailbox_url (char **pout, const char *name)
{
  int rc;
  const char *pat = mu_mailbox_url ();
  mu_vartab_t vtab;

  mu_vartab_create (&vtab);
  mu_vartab_define (vtab, "user", name, 1);
  rc = mu_vartab_expand (vtab, pat, pout);
  mu_vartab_destroy (&vtab);
  return rc;
}

/* Is this a security risk?  */
#define USE_ENVIRON 1

static int
split_shortcut (const char *file, const char pfx[], char **user, char **rest)
{
  *user = NULL;
  *rest = NULL;

  if (!strchr (pfx, file[0]))
    return 0;

  if (*++file == 0)
    return 0;
  else
    {
      char *p = strchr (file, '/');
      int len;
      if (p)
        len = p - file + 1;
      else
        len = strlen (file) + 1;

      if (len == 1)
	*user = NULL;
      else
	{
	  *user = calloc (1, len);
	  if (!*user)
	    return ENOMEM;

	  memcpy (*user, file, len);
	  (*user)[len-1] = 0;
	}
      file += len-1;
      if (file[0] == '/')
        file++;
    }

  if (file[0])
    {
      *rest = strdup (file);
      if (!*rest)
        {
          free (*user);
          return ENOMEM;
        }
    }
  
  return 0;
}

static char *
get_homedir (const char *user)
{
  char *homedir = NULL;
  struct mu_auth_data *auth = NULL;
  
  if (user)
    {
      auth = mu_get_auth_by_name (user);
      if (auth)
        homedir = auth->dir;
    }
  else
    {
#ifdef USE_ENVIRON
      /* NOTE: Should we honor ${HOME}?  */
      homedir = getenv ("HOME");
      if (homedir == NULL)
        {
	  auth = mu_get_auth_by_name (user);
	  if (auth)
	    homedir = auth->dir;
        }
#else
      auth = mu_get_auth_by_name (user);
      if (auth)
	homedir = auth->dir;
#endif
    }

  if (homedir)
    homedir = strdup (homedir);
  mu_auth_data_free (auth);
  return homedir;
}

static int
user_mailbox_name (const char *user, char **mailbox_name)
{
#ifdef USE_ENVIRON
  if (!user)
    user = (getenv ("LOGNAME")) ? getenv ("LOGNAME") : getenv ("USER");
#endif

  if (user)
    {
      int rc = mu_construct_user_mailbox_url (mailbox_name, user);
      if (rc)
	return rc;
    }
  else
    {
      struct mu_auth_data *auth = mu_get_auth_by_uid (getuid ());

      if (!auth)
        {
          mu_error ("Who am I?");
          return EINVAL;
        }
      *mailbox_name = strdup (auth->mailbox);
      mu_auth_data_free (auth);
    }

  return 0;
}

static int
plus_expand (const char *file, char **buf)
{
  char *home;
  const char *folder_dir = mu_folder_directory ();
  int len;

  home = get_homedir (NULL);
  if (!home)
    return ENOENT;
  
  file++;
  
  if (folder_dir[0] == '/' || mu_is_proto (folder_dir))
    {
      len = strlen (folder_dir) + strlen (file) + 2;
      *buf = malloc (len);
      sprintf (*buf, "%s/%s", folder_dir, file);
    }
  else
    {
      len = strlen (home) + strlen (folder_dir) + strlen (file) + 3;
      *buf = malloc (len);
      sprintf (*buf, "%s/%s/%s", home, folder_dir, file);
    }
  (*buf)[len-1] = 0;
  
  free (home);
  return 0;
}

static int
percent_expand (const char *file, char **mbox)
{
  char *user = NULL;
  char *path = NULL;
  int status;
  
  if ((status = split_shortcut (file, "%", &user, &path)))
    return status;

  if (path)
    {
      free (user);
      free (path);
      return ENOENT;
    }

  status = user_mailbox_name (user, mbox);
  free (user);
  return status;
}

static void
attach_auth_ticket (mu_mailbox_t mbox)
{
  mu_folder_t folder = NULL;
  mu_authority_t auth = NULL;

  if (mu_mailbox_get_folder (mbox, &folder) == 0
      && mu_folder_get_authority (folder, &auth) == 0
      && auth)
    {
      char *filename = mu_tilde_expansion (mu_ticket_file, "/", NULL);
      mu_wicket_t wicket;
      int rc;
  
      MU_DEBUG1 (mbox->debug, MU_DEBUG_TRACE1,
		 "Reading user ticket file %s\n", filename);
      if ((rc = mu_file_wicket_create (&wicket, filename)) == 0)
	{
	  mu_ticket_t ticket;
      
	  if ((rc = mu_wicket_get_ticket (wicket, NULL, &ticket)) == 0)
	    {
	      rc = mu_authority_set_ticket (auth, ticket);
	      MU_DEBUG1 (mbox->debug, MU_DEBUG_TRACE1,
			 "Retrieved and set ticket: %d\n", rc);
	    }
	  else
	    MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR,
		       "Error retrieving ticket: %s\n",
		       mu_strerror (rc));
	  mu_wicket_destroy (&wicket);
	}
      else
	MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR,
		   "Error creating wicket: %s\n", mu_strerror (rc));
      free (filename);
    }
}

/* We are trying to be smart about the location of the mail.
   mu_mailbox_create() is not doing this.
   %           --> system mailbox for the real uid
   %user       --> system mailbox for the given user
   ~/file      --> /home/user/file
   ~user/file  --> /home/user/file
   +file       --> /home/user/Mail/file
   =file       --> /home/user/Mail/file
*/
int
mu_mailbox_create_default (mu_mailbox_t *pmbox, const char *mail)
{
  char *mbox = NULL;
  char *tmp_mbox = NULL;
  char *p;
  int status = 0;

  /* Sanity.  */
  if (pmbox == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (mail && *mail == 0)
    mail = NULL;
  
  if (mail == NULL)
    {
      if (!_mu_mailbox_pattern)
	{
	  /* Other utilities may not understand GNU mailutils url namespace, so
	     use FOLDER instead, to not confuse others by using MAIL.  */
	  mail = getenv ("FOLDER");
	  if (!mail)
	    {
	      /* Fallback to well-known environment.  */
	      mail = getenv ("MAIL");
	    }
	}

      if (!mail)
	{
	  if ((status = user_mailbox_name (NULL, &tmp_mbox)))
	    return status;
	  mail = tmp_mbox;
	}
    }

  p = mu_tilde_expansion (mail, "/", NULL);
  if (tmp_mbox)
    free (tmp_mbox);
  tmp_mbox = p;
  mail = tmp_mbox;
  if (!mail)
    return ENOMEM;
  
  switch (mail[0])
    {
    case '%':
      status = percent_expand (mail, &mbox);
      break;
      
    case '+':
    case '=':
      status = plus_expand (mail, &mbox);
      break;

    case '/':
      mbox = strdup (mail);
      break;
      
    default:
      if (!mu_is_proto (mail))
	{
	  p = mu_getcwd();
	  mbox = malloc (strlen (p) + strlen (mail) + 2);
	  sprintf (mbox, "%s/%s", p, mail);
	  free (p);  
	}
      else
	mbox = strdup (mail);
      break;
    }

  if (tmp_mbox)
    free (tmp_mbox);

  if (status)
    return status;
  
  status = mu_mailbox_create (pmbox, mbox);
  free (mbox);
  if (status == 0)
    attach_auth_ticket (*pmbox);
      
  return status;
}
