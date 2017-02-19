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
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include <mailutils/stream.h>

#if defined(WITH_GDBM)

#include <gdbm.h>
#define USE_DBM
typedef GDBM_FILE DBM_FILE;
typedef datum DBM_DATUM;
#define MU_DATUM_SIZE(d) (d).dsize
#define MU_DATUM_PTR(d) (d).dptr

#elif defined(WITH_BDB)

#include <db.h>
#define USE_DBM

struct db2_file
{
  DB *db;
  DBC *dbc;
};

typedef struct db2_file *DBM_FILE;
typedef DBT DBM_DATUM;
#define MU_DATUM_SIZE(d) (d).size
#define MU_DATUM_PTR(d) (d).data

#elif defined(WITH_NDBM)

#include <ndbm.h>
#define USE_DBM
typedef DBM *DBM_FILE;
typedef datum DBM_DATUM;
#define MU_DATUM_SIZE(d) (d).dsize
#define MU_DATUM_PTR(d) (d).dptr

#elif defined(WITH_TOKYOCABINET)

#include <tcutil.h>
#include <tchdb.h>
#define USE_DBM

struct tokyocabinet_file
{
  TCHDB *hdb;
};

struct tokyocabinet_datum {
  void *data;
  int size;
};

typedef struct tokyocabinet_file *DBM_FILE;
typedef struct tokyocabinet_datum DBM_DATUM;
#define MU_DATUM_SIZE(d) (d).size
#define MU_DATUM_PTR(d) (d).data

#endif

#ifdef USE_DBM
struct stat;
int mu_dbm_stat (char *name, struct stat *sb);
int mu_dbm_open (char *name, DBM_FILE *db, int flags, int mode);
int mu_dbm_close (DBM_FILE db);
int mu_dbm_fetch (DBM_FILE db, DBM_DATUM key, DBM_DATUM *ret);
int mu_dbm_insert (DBM_FILE db, DBM_DATUM key, DBM_DATUM contents, int replace);
int mu_dbm_delete (DBM_FILE db, DBM_DATUM key);
DBM_DATUM mu_dbm_firstkey (DBM_FILE db);
DBM_DATUM mu_dbm_nextkey (DBM_FILE db, DBM_DATUM key);
void mu_dbm_datum_free(DBM_DATUM *datum);
#endif /* USE_DBM */

int mu_fcheck_perm (int fd, int mode);
int mu_check_perm (const char *name, int mode);
