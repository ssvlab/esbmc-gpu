/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "pop3d.h"

/*
  APOP name digest

  Arguments:
  a string identifying a mailbox and a MD5 digest string
  (both required)

  Restrictions:
  may only be given in the AUTHORIZATION state after the POP3
  greeting or after an unsuccessful USER or PASS command

  When the POP3 server receives the APOP command, it verifies
  the digest provided.  If the digest is correct, the POP3
  server issues a positive response, and the POP3 session
  enters the TRANSACTION state.  Otherwise, a negative
  response is issued and the POP3 session remains in the
  AUTHORIZATION state.  */

/* Check if a username exists in APOP password file
   returns pointer to password if found, otherwise NULL */

char *
pop3d_apopuser (const char *user)
{
  char *password;
  char buf[POP_MAXCMDLEN];

#ifdef USE_DBM
  {
    DBM_FILE db;
    DBM_DATUM key, data;

    int rc = mu_dbm_open (APOP_PASSFILE, &db, MU_STREAM_READ, 0600);
    if (rc)
      {
	mu_diag_output (MU_DIAG_ERROR, _("unable to open APOP db: %s"),
		mu_strerror (errno));
	return NULL;
      }

    memset (&key, 0, sizeof key);
    memset (&data, 0, sizeof data);

    strncpy (buf, user, sizeof buf);
    /* strncpy () is lame and does not NULL terminate.  */
    buf[sizeof (buf) - 1] = '\0';
    MU_DATUM_PTR(key) = buf;
    MU_DATUM_SIZE(key) = strlen (buf);

    rc = mu_dbm_fetch (db, key, &data);
    mu_dbm_close (db);
    if (rc)
      {
	mu_diag_output (MU_DIAG_ERROR,
			_("cannot fetch APOP data: %s"), mu_strerror (errno));
	return NULL;
      }
    password = calloc (MU_DATUM_SIZE(data) + 1, sizeof (*password));
    if (password == NULL)
      {
	mu_dbm_datum_free (&data);
	return NULL;
      }
    
    sprintf (password, "%.*s", (int) MU_DATUM_SIZE(data),
	     (char*) MU_DATUM_PTR(data));
    mu_dbm_datum_free (&data);
    return password;
  }
#else /* !USE_DBM */
  {
    char *tmp;
    FILE *apop_file;

    if (mu_check_perm (APOP_PASSFILE, 0600))
      {
	mu_diag_output (MU_DIAG_INFO,
			_("bad permissions on APOP password file"));
	return NULL;
    }

    apop_file = fopen (APOP_PASSFILE, "r");
    if (apop_file == NULL)
      {
	mu_diag_output (MU_DIAG_INFO, _("unable to open APOP password file %s"),
		strerror (errno));
	return NULL;
      }

    password = calloc (APOP_DIGEST, sizeof (*password));
    if (password == NULL)
      {
	fclose (apop_file);
	pop3d_abquit (ERR_NO_MEM);
      }

    while (fgets (buf, sizeof (buf) - 1, apop_file) != NULL)
      {
	tmp = strchr (buf, ':');
	if (tmp == NULL)
	  continue;
	*tmp++ = '\0';

	if (strncmp (user, buf, strlen (user)))
	  continue;

	strncpy (password, tmp, APOP_DIGEST);
	/* strncpy () is lame and does not NULL terminate.  */
	password[APOP_DIGEST - 1] = '\0';
	tmp = strchr (password, '\n');
	if (tmp)
	  *tmp = '\0';
	break;
      }

    fclose (apop_file);
    if (*password == '\0')
      {
	free (password);
	return NULL;
      }

    return password;
  }
#endif
}

int
pop3d_apop (char *arg)
{
  char *tmp, *password, *user_digest, *user;
  char buf[POP_MAXCMDLEN];
  struct mu_md5_ctx md5context;
  unsigned char md5digest[16];

  if (state != AUTHORIZATION)
    return ERR_WRONG_STATE;

  if (strlen (arg) == 0)
    return ERR_BAD_ARGS;

  pop3d_parse_command (arg, &user, &user_digest);
  if (strlen (user) > (POP_MAXCMDLEN - APOP_DIGEST))
    {
      mu_diag_output (MU_DIAG_INFO, _("user name too long: %s"), user);
      return ERR_BAD_ARGS;
    }

  password = pop3d_apopuser (user);
  if (password == NULL)
    {
      mu_diag_output (MU_DIAG_INFO,
		      _("password for `%s' not found in the database"),
	      user);
      return ERR_BAD_LOGIN;
    }

  mu_md5_init_ctx (&md5context);
  mu_md5_process_bytes (md5shared, strlen (md5shared), &md5context);
  mu_md5_process_bytes (password, strlen (password), &md5context);
  free (password);
  mu_md5_finish_ctx (&md5context, md5digest);

  {
    int i;
    tmp = buf;
    for (i = 0; i < 16; i++, tmp += 2)
      sprintf (tmp, "%02x", md5digest[i]);
  }

  *tmp++ = '\0';

  if (strcmp (user_digest, buf))
    {
      mu_diag_output (MU_DIAG_INFO, _("APOP failed for `%s'"), user);
      return ERR_BAD_LOGIN;
    }

  auth_data = mu_get_auth_by_name (user);
  if (auth_data == NULL)
    return ERR_BAD_LOGIN;

  return pop3d_begin_session ();
}
