/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2007, 2009, 2010 Free Software Foundation, Inc.

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

#include "pop3d.h"

#ifdef ENABLE_LOGIN_DELAY

static int
open_stat_db (DBM_FILE *db, int mode)
{
  int rc = mu_dbm_open (login_stat_file, db, mode, 0660);
  if (rc)
    {
      if (rc == -1)
	mu_diag_output (MU_DIAG_INFO, _("bad permissions on statistics db"));
      else
	mu_diag_output (MU_DIAG_ERROR, _("unable to open statistics db: %s"),
		mu_strerror (rc));
    }
  return rc;
}

int
check_login_delay (char *username)
{
  time_t now, prev_time;
  DBM_FILE db;
  DBM_DATUM key, data;
  char text[64], *p;
  int rc;

  if (login_delay == 0)
    return 0;
  
  time (&now);
  if (open_stat_db (&db, MU_STREAM_RDWR))
    return 0;
  
  memset (&key, 0, sizeof key);
  MU_DATUM_PTR(key) = username;
  MU_DATUM_SIZE(key) = strlen (username);
  memset (&data, 0, sizeof data);

  rc = mu_dbm_fetch (db, key, &data);
  if (rc)
    {
      mu_diag_output (MU_DIAG_ERROR, _("cannot fetch login delay data: %s"),
	      mu_strerror (rc));
      mu_dbm_close (db);
      return 0;
    }

  if (MU_DATUM_SIZE(data) > sizeof (text) - 1)
    {
      mu_diag_output (MU_DIAG_ERROR,
		      _("invalid entry for '%s': wrong timestamp size"),
	      username);
      mu_dbm_close (db);
      return 0;
    }

  memcpy (text, MU_DATUM_PTR(data), MU_DATUM_SIZE(data));
  text[MU_DATUM_SIZE(data)] = 0;
  mu_dbm_close (db);

  prev_time = strtoul (text, &p, 0);
  if (*p)
    {
      mu_diag_output (MU_DIAG_ERROR,
		      _("malformed timestamp for '%s': %s"),
	      username, text);
      return 0;
    }

  return now - prev_time < login_delay;
}

void
update_login_delay (char *username)
{
  time_t now;
  DBM_FILE db;
  DBM_DATUM key, data;
  char text[64];
  
  if (login_delay == 0 || username == NULL)
    return;

  time (&now);
  if (open_stat_db (&db, MU_STREAM_RDWR))
    return;
  
  memset(&key, 0, sizeof(key));
  memset(&data, 0, sizeof(data));
  MU_DATUM_PTR(key) = username;
  MU_DATUM_SIZE(key) = strlen (username);
  snprintf (text, sizeof text, "%lu", (unsigned long) now);
  MU_DATUM_PTR(data) = text;
  MU_DATUM_SIZE(data) = strlen (text);
  if (mu_dbm_insert (db, key, data, 1))
    mu_error (_("%s: cannot store datum %s/%s"),
	      login_stat_file, username, text);
  
  mu_dbm_close (db);
}

void
login_delay_capa ()
{
  DBM_FILE db;
  
  if (login_delay && open_stat_db (&db, MU_STREAM_RDWR) == 0)
    {
      pop3d_outf ("LOGIN-DELAY %s\r\n", mu_umaxtostr (0, login_delay));
      mu_dbm_close (db);
    }
}
#endif
