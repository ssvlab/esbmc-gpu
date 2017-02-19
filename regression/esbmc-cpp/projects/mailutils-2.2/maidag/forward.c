/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

/* ".forward" support for GNU Maidag */

#include "maidag.h"

/* Functions for checking file mode of .forward and its directory.
   Each of these checks certain bits and returns 0 if they are OK
   and non-0 otherwise. */

static int
check_iwgrp (struct stat *filest, struct stat *dirst)
{
  return filest->st_mode & S_IWGRP;
}

static int
check_iwoth (struct stat *filest, struct stat *dirst)
{
  return filest->st_mode & S_IWOTH;
}

static int
check_linked_wrdir (struct stat *filest, struct stat *dirst)
{
  return (filest->st_mode & S_IFLNK) && (dirst->st_mode & (S_IWGRP | S_IWOTH));
}

static int
check_dir_iwgrp (struct stat *filest, struct stat *dirst)
{
  return dirst->st_mode & S_IWGRP;
}

static int
check_dir_iwoth (struct stat *filest, struct stat *dirst)
{
  return dirst->st_mode & S_IWOTH;
}

/* The table of permission checkers below has this type: */
struct perm_checker
{
  int flag;              /* FWD_ flag that enables this entry */
  char *descr;           /* Textual description to use if FUN returns !0 */
  int (*fun) (struct stat *filest, struct stat *dirst); /* Checker function */
};

static struct perm_checker perm_check_tab[] = {
  { FWD_IWGRP, N_("group writable forward file"), check_iwgrp },
  { FWD_IWOTH, N_("world writable forward file"), check_iwoth },
  { FWD_LINK, N_("linked forward file in writable dir"), check_linked_wrdir },
  { FWD_DIR_IWGRP, N_("forward file in group writable directory"),
    check_dir_iwgrp },
  { FWD_DIR_IWOTH, N_("forward file in world writable directory"),
    check_dir_iwoth },
  { 0 }
};

/* Check if the forwrd file FILENAME has right permissions and file mode.
   DIRST describes the directory holding the file, AUTH gives current user
   authority. */
int
check_forward_permissions (const char *filename, struct stat *dirst,
			   struct mu_auth_data *auth)
{
  struct stat st;
  
  if (stat (filename, &st) == 0)
    {
      int i;
      
      if (auth->uid != st.st_uid)
	{
	  mu_error (_("%s not owned by %s"), filename, auth->name);
	  return 1;
	}
      for (i = 0; perm_check_tab[i].flag; i++)
	if ((forward_file_checks & perm_check_tab[i].flag)
	    && perm_check_tab[i].fun (&st, dirst))
	  {
	    mu_error ("%s: %s", filename, gettext (perm_check_tab[i].descr));
	    return 1;
	  }
      return 0;
    }
  else if (errno != ENOENT)
    mu_error (_("%s: cannot stat file: %s"),
	      filename, mu_strerror (errno));
  return 1;
}


/* Auxiliary functions */

/* Forward message MSG to given EMAIL, using MAILER and sender address FROM */
static int
forward_to_email (mu_message_t msg, mu_address_t from,
		  mu_mailer_t mailer, const char *email)
{
  mu_address_t to;
  int rc;
  
  rc = mu_address_create (&to, email);
  if (rc)
    {
      mu_error (_("%s: cannot create email: %s"), email, mu_strerror (rc));
      return 1;
    }

  rc = mu_mailer_send_message (mailer, msg, from, to);
  if (rc)
    mu_error (_("Sending message to `%s' failed: %s"),
	      email, mu_strerror (rc));
  mu_address_destroy (&to);
  return rc;
}

/* Create a mailer if it does not already exist.*/
int
forward_mailer_create (mu_mailer_t *pmailer)
{
  int rc;

  if (*pmailer == NULL)
    {
      rc = mu_mailer_create (pmailer, NULL);
      if (rc)
	{
	  const char *url = NULL;
	  mu_mailer_get_url_default (&url);
	  mu_error (_("Creating mailer `%s' failed: %s"),
		    url, mu_strerror (rc));
	  return 1;
	}

  
      rc = mu_mailer_open (*pmailer, 0);
      if (rc)
	{
	  const char *url = NULL;
	  mu_mailer_get_url_default (&url);
	  mu_error (_("Opening mailer `%s' failed: %s"),
		    url, mu_strerror (rc));
	  mu_mailer_destroy (pmailer);
	  return 1;
	}
    }
  return 0;
}

/* Create *PFROM (if it is NULL), from the envelope sender address of MSG. */
static int
create_from_address (mu_message_t msg, mu_address_t *pfrom)
{
  if (!*pfrom)
    {
      mu_envelope_t envelope;
      const char *str;
      int status = mu_message_get_envelope (msg, &envelope);
      if (status)
	{
	  mu_error (_("cannot get envelope: %s"), mu_strerror (status));
	  return 1;
	}
      status = mu_envelope_sget_sender (envelope, &str);
      if (status)
	{
	  mu_error (_("cannot get envelope sender: %s"), mu_strerror (status));
	  return 1;
	}
      status = mu_address_create (pfrom, str);
      if (status)
	{
	  mu_error (_("%s: cannot create email: %s"), str,
		    mu_strerror (status));
	  return 1;
	}
    }
  return 0;
}       


/* Forward message MSG as requested by file FILENAME.
   MYNAME gives local user name. */
enum maidag_forward_result
process_forward (mu_message_t msg, char *filename, const char *myname)
{
  FILE *fp;
  size_t size = 0;
  char *buf = NULL;
  enum maidag_forward_result result = maidag_forward_ok;
  mu_mailer_t mailer = NULL;
  mu_address_t from = NULL;
  
  fp = fopen (filename, "r");
  if (!fp)
    {
      mu_error (_("%s: cannot open forward file: %s"),
		filename, mu_strerror (errno));
      return maidag_forward_error;
    }

  while (getline (&buf, &size, fp) > 0)
    {
      char *p;

      mu_rtrim_class (buf, MU_CTYPE_SPACE);
      p = mu_str_skip_class (buf, MU_CTYPE_SPACE);

      if (*p && *p != '#')
	{
	  if (strchr (p, '@'))
	    {
	      if (create_from_address (msg, &from)
		  || forward_mailer_create (&mailer)
		  || forward_to_email (msg, from, mailer, p))
		result = maidag_forward_error;
	    }
	  else 
	    {
	      if (*p == '\\')
		p++;
	      if (strcmp (p, myname) == 0)
		{
		  if (result == maidag_forward_ok)
		    result = maidag_forward_metoo;
		}
	      else if (deliver (msg, p, NULL))
		result = maidag_forward_error;
	    }
	}
    }

  mu_address_destroy (&from);
  if (mailer)
    {
      mu_mailer_close (mailer);
      mu_mailer_destroy (&mailer);
    }
  free (buf);
  fclose (fp);
  return result;
}

/* Check if the forward file FWFILE for user given by AUTH exists, and if
   so, use to to forward message MSG. */
enum maidag_forward_result
maidag_forward (mu_message_t msg, struct mu_auth_data *auth, char *fwfile)
{
  struct stat st;
  char *filename;
  enum maidag_forward_result result = maidag_forward_none;
  
  if (stat (auth->dir, &st))
    {
      if (errno == ENOENT)
	/* FIXME: a warning, maybe? */;
      else if (!S_ISDIR (st.st_mode))
	mu_error (_("%s: not a directory"), auth->dir);
      else
	mu_error (_("%s: cannot stat directory: %s"),
		  auth->dir, mu_strerror (errno));
      return maidag_forward_none;
    }
  asprintf (&filename, "%s/%s", auth->dir, fwfile);
  if (!filename)
    {
      mu_error ("%s", mu_strerror (errno));
      return maidag_forward_error;
    }

  if (check_forward_permissions (filename, &st, auth) == 0)
    result = process_forward (msg, filename, auth->name);
  
  free (filename);
  return result;
}

