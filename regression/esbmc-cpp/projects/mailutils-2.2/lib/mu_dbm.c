/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2006, 2007, 2009, 2010 Free
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
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#include <errno.h>
#include <mailutils/errno.h>
#include <mu_dbm.h>
#include <xalloc.h>

int
mu_fcheck_perm (int fd, int mode)
{
  struct stat st;

  if (mode == 0)
    return 0;
  if (fstat (fd, &st) == -1)
    {
      if (errno == ENOENT)
	return 0;
      else
	return 1;
    }
  if ((st.st_mode & 0777) != mode)
    {
      errno = MU_ERR_UNSAFE_PERMS;
      return 1;
    }
  return 0;
}

int
mu_check_perm (const char *name, int mode)
{
  struct stat st;

  if (mode == 0)
    return 0;
  if (stat (name, &st) == -1)
    {
      if (errno == ENOENT)
	return 0;
      else
	return 1;
    }
  if ((st.st_mode & 0777) != mode)
    {
      errno = MU_ERR_UNSAFE_PERMS;
      return 1;
    }
  return 0;
}

static char *
make_db_name (const char *name, const char *suffix)
{
  int nlen = strlen (name);
  int slen = strlen (suffix);
  char *p;
  
  if (nlen > slen && strcmp (name + nlen - slen, suffix) == 0)
    p = xstrdup (name);
  else
    {
      p = xmalloc (strlen (name) + slen + 1);
      strcat (strcpy (p, name), suffix);
    }
  return p;
}

#if defined(WITH_GDBM)

#define DB_SUFFIX ".db"

int
mu_dbm_stat (char *name, struct stat *sb)
{
  int rc;
  char *pfname = make_db_name (name, DB_SUFFIX);
  rc = stat (pfname, sb);
  free (pfname);
  return rc;
}

int
mu_dbm_open (char *name, DBM_FILE *db, int flags, int mode)
{
  int f;
  char *pfname = make_db_name (name, DB_SUFFIX);

  if (mu_check_perm (pfname, mode))
    {
      free (pfname);
      return -1;
    }

  switch (flags)
    {
    case MU_STREAM_CREAT:
      f = GDBM_NEWDB;
      break;
      
    case MU_STREAM_READ:
      f = GDBM_READER;
      break;
      
    case MU_STREAM_RDWR:
      f = GDBM_WRCREAT;
      break;
      
    default:
      free (pfname);
      errno = EINVAL;
      return 1;
    }
  *db = gdbm_open (pfname, 512, f, mode, NULL);
  free (pfname);
  return *db == NULL;
}

int
mu_dbm_close (DBM_FILE db)
{
  gdbm_close(db);
  return 0;
}

int
mu_dbm_fetch (DBM_FILE db, DBM_DATUM key, DBM_DATUM *ret)
{
  *ret = gdbm_fetch (db, key);
  return ret->dptr == NULL;
}

int
mu_dbm_delete (DBM_FILE db, DBM_DATUM key)
{
  return gdbm_delete (db, key);
}

int
mu_dbm_insert (DBM_FILE db, DBM_DATUM key, DBM_DATUM contents, int replace)
{
  return gdbm_store (db, key, contents,
		     replace ? GDBM_REPLACE : GDBM_INSERT);
}

DBM_DATUM
mu_dbm_firstkey (DBM_FILE db)
{
  return gdbm_firstkey (db);
}

DBM_DATUM
mu_dbm_nextkey (DBM_FILE db, DBM_DATUM key)
{
  return gdbm_nextkey (db, key);
}

void
mu_dbm_datum_free (DBM_DATUM *datum)
{
  void *ptr = MU_DATUM_PTR (*datum);
  if (ptr)
    free (ptr);
  MU_DATUM_PTR (*datum) = 0;
}

#elif defined(WITH_BDB)

#define DB_SUFFIX ".db"

int
mu_dbm_stat (char *name, struct stat *sb)
{
  int rc;
  char *pfname = make_db_name (name, DB_SUFFIX);
  rc = stat (pfname, sb);
  free (pfname);
  return rc;
}

int
mu_dbm_open (char *name, DBM_FILE *dbm, int flags, int mode)
{
  int f, rc;
  DB *db;
  char *pfname = make_db_name (name, DB_SUFFIX);

  if (mu_check_perm (pfname, mode))
    {
      free (pfname);
      errno = MU_ERR_UNSAFE_PERMS;
      return -1;
    }

  switch (flags)
    {
    case MU_STREAM_CREAT:
      f = DB_CREATE|DB_TRUNCATE;
      break;
      
    case MU_STREAM_READ:
      f = DB_RDONLY;
      break;
      
    case MU_STREAM_RDWR:
      f = DB_CREATE;
      break;
      
    default:
      free (pfname);
      errno = EINVAL;
      return -1;
    }

#if WITH_BDB == 2  
  rc = db_open (pfname, DB_HASH, f, mode, NULL, NULL, &db);
#else
  rc = db_create (&db, NULL, 0);
  if (rc != 0 || db == NULL)
    return rc;
# if DB_VERSION_MAJOR == 3
  rc = db->open (db, pfname, NULL, DB_HASH, f, mode);
# else
  rc = db->open (db, NULL, pfname, NULL, DB_HASH, f, mode);
# endif
#endif
  
  free (pfname);
  if (rc)
    return -1;

  *dbm = malloc (sizeof **dbm);
  if (!*dbm)
    {
      db->close (db, 0);
      errno = ENOMEM;
      return -1;
    }
  (*dbm)->db = db;
  (*dbm)->dbc = NULL;
  return 0;
}

int
mu_dbm_close (DBM_FILE db)
{
  db->db->close (db->db, 0);
  free (db);
  return 0;
}

int
mu_dbm_fetch (DBM_FILE db, DBM_DATUM key, DBM_DATUM *ret)
{
  return db->db->get (db->db, NULL, &key, ret, 0);
}

int
mu_dbm_delete (DBM_FILE db, DBM_DATUM key)
{
  return db->db->del (db->db, NULL, &key, 0);
}

int
mu_dbm_insert (DBM_FILE db, DBM_DATUM key, DBM_DATUM contents, int replace)
{
  /*FIXME: replace unused*/
  return db->db->put (db->db, NULL, &key, &contents, 0);
}

DBM_DATUM
mu_dbm_firstkey (DBM_FILE db)
{
  DBT key, data;
  int ret;

  memset (&key, 0, sizeof key);
  memset (&data, 0, sizeof data);

  if (!db->dbc)
    {
      if (db->db->cursor (db->db, NULL, &db->dbc BDB2_CURSOR_LASTARG) != 0)
	return key;
    }

  if ((ret = db->dbc->c_get (db->dbc, &key, &data, DB_FIRST)) != 0)
    {
      key.data = NULL;
      key.size = 0;
      if (ret == DB_NOTFOUND)
	errno = MU_ERR_NOENT;
      else
	errno = ret;
    }
  return key;
}

DBM_DATUM
mu_dbm_nextkey (DBM_FILE db, DBM_DATUM pkey /*unused*/)
{
  DBT key, data;
  int ret;

  memset (&key, 0, sizeof key);
  memset (&data, 0, sizeof data);

  if (!db->dbc)
    return key;

  if ((ret = db->dbc->c_get (db->dbc, &key, &data, DB_NEXT)) != 0)
    {
      key.data = NULL;
      key.size = 0;
      if (ret == DB_NOTFOUND)
	errno = MU_ERR_NOENT;
      else
	errno = ret;
    }
  return key;
}

void
mu_dbm_datum_free (DBM_DATUM *datum)
{
  /* empty */
}

#elif defined(WITH_NDBM)

#define DB_SUFFIX ".pag"

int
mu_dbm_stat (char *name, struct stat *sb)
{
  int rc;
  char *pfname = make_db_name (name, DB_SUFFIX);
  rc = stat (pfname, sb);
  free (pfname);
  return rc;
}

int
mu_dbm_open (char *name, DBM_FILE *db, int flags, int mode)
{
  int f;
  char *pfname;

  switch (flags)
    {
    case MU_STREAM_CREAT:
      f = O_CREAT|O_TRUNC|O_RDWR;
      break;
      
    case MU_STREAM_READ:
      f = O_RDONLY;
      break;
      
    case MU_STREAM_RDWR:
      f = O_CREAT|O_RDWR;
      break;
      
    default:
      errno = EINVAL;
      return -1;
    }
  pfname = strip_suffix (name, DB_SUFFIX);
  *db = dbm_open (pfname, f, mode);
  free (pfname);
  if (!*db)
    return -1;

  if (mu_fcheck_perm (dbm_dirfno (*db), mode)
      || mu_fcheck_perm (dbm_pagfno (*db), mode))
    {
      dbm_close (*db);
      return 1;
    }

  return 0;
}

int
mu_dbm_close (DBM_FILE db)
{
  dbm_close (db);
  return 0;
}

int
mu_dbm_fetch (DBM_FILE db, DBM_DATUM key, DBM_DATUM *ret)
{
  *ret = dbm_fetch (db, key);
  return ret->dptr == NULL;
}

int
mu_dbm_delete (DBM_FILE db, DBM_DATUM key)
{
  return dbm_delete (db, key);
}

int
mu_dbm_insert (DBM_FILE db, DBM_DATUM key, DBM_DATUM contents, int replace)
{
  return dbm_store (db, key, contents, replace ? DBM_REPLACE : DBM_INSERT);
}

DBM_DATUM
mu_dbm_firstkey (DBM_FILE db)
{
  return dbm_firstkey (db);
}

DBM_DATUM
mu_dbm_nextkey (DBM_FILE db, DBM_DATUM key)
{
  return dbm_nextkey (db, key);
}

void
mu_dbm_datum_free (DBM_DATUM *datum)
{
  /* empty */
}

#elif defined(WITH_TOKYOCABINET)

#define DB_SUFFIX ".tch"

int
mu_dbm_stat (char *name, struct stat *sb)
{
  int rc;
  char *pfname = make_db_name (name, DB_SUFFIX);
  rc = stat (pfname, sb);
  free (pfname);
  return rc;
}

int
mu_dbm_open (char *name, DBM_FILE *db, int flags, int mode)
{
  int f, ecode;
  char *pfname = make_db_name (name, DB_SUFFIX);

  if (mu_check_perm (pfname, mode))
    {
      free (pfname);
      return -1;
    }

  switch (flags)
    {
    case MU_STREAM_CREAT:
      f = HDBOWRITER | HDBOCREAT;
      break;
      
    case MU_STREAM_READ:
      f = HDBOREADER;
      break;
      
    case MU_STREAM_RDWR:
      f = HDBOREADER | HDBOWRITER;
      break;
      
    default:
      free (pfname);
      errno = EINVAL;
      return 1;
    }

  *db = malloc (sizeof **db);
  if (!*db)
    {
      errno = ENOMEM;
      return -1;
    }
  (*db)->hdb = tchdbnew ();

  if (!tchdbopen ((*db)->hdb, pfname, f))
    ecode = tchdbecode ((*db)->hdb);

  free (pfname);
  return 0;
}

int
mu_dbm_close (DBM_FILE db)
{
  tchdbclose (db->hdb);
  tchdbdel (db->hdb);
  return 0;
}

int
mu_dbm_fetch (DBM_FILE db, DBM_DATUM key, DBM_DATUM *ret)
{
  ret->data = tchdbget (db->hdb, key.data, key.size, &ret->size);
  return ret->data == NULL;
}

int
mu_dbm_delete (DBM_FILE db, DBM_DATUM key)
{
  return !tchdbout (db->hdb, key.data, key.size);
}

int
mu_dbm_insert (DBM_FILE db, DBM_DATUM key, DBM_DATUM contents, int replace)
{
  if (replace)
    return !tchdbput (db->hdb, key.data, key.size, contents.data, contents.size);
  else
    return !tchdbputkeep (db->hdb, key.data, key.size,
			  contents.data, contents.size);
}

DBM_DATUM
mu_dbm_firstkey (DBM_FILE db)
{
  DBM_DATUM key;
  memset (&key, 0, sizeof key);

  tchdbiterinit (db->hdb);
  key.data = tchdbiternext (db->hdb, &key.size);
  return key;
}

DBM_DATUM
mu_dbm_nextkey (DBM_FILE db, DBM_DATUM unused)
{
  DBM_DATUM key;
  memset (&key, 0, sizeof key);

  key.data = tchdbiternext (db->hdb, &key.size);
  return key;
}

void
mu_dbm_datum_free (DBM_DATUM *datum)
{
  /* empty */
}

#endif

