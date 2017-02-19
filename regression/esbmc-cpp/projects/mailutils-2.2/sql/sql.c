/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2006, 2007, 2009, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <mailutils/mailutils.h>
#include <mailutils/sql.h>
#include "modlist.h"

#define NSTATIC_MODS sizeof(static_dispatch_tab)/sizeof(static_dispatch_tab[0])

static mu_sql_dispatch_t **sql_disptab;
size_t sql_disptab_next;
size_t sql_disptab_size;

static int
init_disptab ()
{
  if (!sql_disptab)
    {
      size_t size;

      sql_disptab_size = NSTATIC_MODS;
      size = sql_disptab_size * sizeof sql_disptab[0];
      sql_disptab = malloc (size);
      if (!sql_disptab)
	return ENOMEM;
      memcpy(sql_disptab, static_dispatch_tab, size);
      sql_disptab_next = sql_disptab_size;
    }
  return 0;
}

/* FIXME: See comment 'For dynamic loading' below */
#if 0
static int
add_disptab (mu_sql_dispatch_t *tab)
{
  if (sql_disptab_next == sql_disptab_size)
    {
      mu_sql_dispatch_t **tmp;

      tmp = realloc (sql_disptab, sql_disptab_size + 4);
      if (!tmp)
	return ENOMEM;
      sql_disptab = tmp;
      sql_disptab_size += 4;
    }
  sql_disptab[sql_disptab_next] = tab;
  return sql_disptab_next++;
}
#endif

int
mu_sql_interface_index (char *name)
{
  int i;
  //mu_sql_dispatch_t *tab;

  init_disptab ();
  for (i = 1; i < sql_disptab_next; i++)
    if (sql_disptab[i] && (!name || strcmp (sql_disptab[i]->name, name) == 0))
      return i;
  /* FIXME: For dynamic loading
      if (name && mu_sql_load_ext (name, "dispatch_tab", &tab))
       return add_disptab (tab);
  */
  return 0;
}

static mu_sql_dispatch_t *
get_sql_entry (int type)
{
  init_disptab ();
  if (type == 0 && sql_disptab[0] == NULL)
    {
      int i;
      for (i = 1; i < sql_disptab_next; i++)
	if (sql_disptab[i])
	  {
	    sql_disptab[type] = sql_disptab[i];
	    break;
	  }
    }
  if (!sql_disptab[type])
    {
      mu_error (_("SQL dispatcher table empty"));
      abort ();
    }
  return sql_disptab[type];
}

#define SQL_F(c,f) (*get_sql_entry ((c)->interface) -> f)

int
mu_sql_connection_init (mu_sql_connection_t *pconn, int interface,
			char *server, int  port, char *login,
			char *password, char *dbname)
{
  static mu_sql_dispatch_t *tab;
  mu_sql_connection_t conn;

  tab = get_sql_entry (interface);
  if (!tab)
    return MU_ERR_NO_INTERFACE;

  conn = calloc (1, sizeof (*conn));
  if (!conn)
    return ENOMEM;
  conn->state = mu_sql_not_connected;
  conn->interface = interface;
  conn->server = server;
  conn->port = port;
  conn->login = login;
  conn->password = password;
  conn->dbname = dbname;
  if (tab->init)
    {
      int rc = tab->init (conn);
      if (rc)
	{
	  free (conn);
	  return rc;
	}
    }
  *pconn = conn;
  return 0;
}

int
mu_sql_connection_destroy (mu_sql_connection_t *conn)
{
  if (!conn || !*conn)
    return EINVAL;

  mu_sql_disconnect (*conn);
  SQL_F (*conn, destroy) (*conn);
  free (*conn);
  *conn = NULL;
  return 0;
}

int
mu_sql_connect (mu_sql_connection_t conn)
{
  int rc;

  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      break;

    case mu_sql_connected:
    case mu_sql_query_run:
      return MU_ERR_DB_ALREADY_CONNECTED;

    case mu_sql_result_available:
      return MU_ERR_RESULT_NOT_RELEASED;
    }
  rc = SQL_F (conn, connect) (conn);
  if (!rc)
    conn->state = mu_sql_connected;
  return rc;
}

int
mu_sql_disconnect (mu_sql_connection_t conn)
{
  int rc;

  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return 0;

    case mu_sql_connected:
    case mu_sql_query_run:
      break;

    case mu_sql_result_available:
      return MU_ERR_RESULT_NOT_RELEASED;
    }
  rc = SQL_F (conn, disconnect) (conn);
  if (rc == 0)
    conn->state = mu_sql_not_connected;
  return rc;
}


int
mu_sql_query (mu_sql_connection_t conn, char *query)
{
  int rc;
  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
    case mu_sql_query_run:
      break;

    case mu_sql_result_available:
      return MU_ERR_RESULT_NOT_RELEASED;
    }

  rc = SQL_F (conn, query) (conn, query);
  if (rc == 0)
    conn->state = mu_sql_query_run;
  return rc;
}

int
mu_sql_store_result (mu_sql_connection_t conn)
{
  int rc;

  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      break;

    case mu_sql_result_available:
      return MU_ERR_RESULT_NOT_RELEASED;
    }

  rc = SQL_F (conn, store_result) (conn);
  if (rc == 0)
    conn->state = mu_sql_result_available;
  return rc;
}

int
mu_sql_release_result (mu_sql_connection_t conn)
{
  int rc;

  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      return MU_ERR_NO_RESULT;

    case mu_sql_result_available:
      break;
    }

  rc = SQL_F (conn, release_result) (conn);
  if (rc == 0)
    conn->state = mu_sql_connected;
  return rc;
}

int
mu_sql_num_tuples (mu_sql_connection_t conn, size_t *np)
{
  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      return MU_ERR_NO_RESULT;

    case mu_sql_result_available:
      break;
    }

  return SQL_F (conn, num_tuples) (conn, np);
}

int
mu_sql_num_columns (mu_sql_connection_t conn, size_t *np)
{
  if (!conn)
    return EINVAL;
  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      return MU_ERR_NO_RESULT;

    case mu_sql_result_available:
      break;
    }
  return SQL_F (conn, num_columns) (conn, np);
}


int
mu_sql_get_column (mu_sql_connection_t conn, size_t nrow, size_t ncol,
		   char **pdata)
{
  if (!conn)
    return EINVAL;
  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      return MU_ERR_NO_RESULT;

    case mu_sql_result_available:
      break;
    }
  return SQL_F (conn, get_column) (conn, nrow, ncol, pdata);
}

int
mu_sql_get_field (mu_sql_connection_t conn, size_t nrow, const char *fname,
		  char **pdata)
{
  int rc;
  size_t fno;

  if (!conn)
    return EINVAL;

  switch (conn->state)
    {
    case mu_sql_not_connected:
      return MU_ERR_DB_NOT_CONNECTED;

    case mu_sql_connected:
      return MU_ERR_NO_QUERY;

    case mu_sql_query_run:
      return MU_ERR_NO_RESULT;

    case mu_sql_result_available:
      break;
    }

  rc = SQL_F (conn, get_field_number) (conn, fname, &fno);
  if (rc == 0)
    rc = SQL_F (conn, get_column) (conn, nrow, fno, pdata);
  return rc;
}

const char *
mu_sql_strerror (mu_sql_connection_t conn)
{
  if (!conn)
    return strerror (EINVAL);
  return SQL_F (conn, errstr) (conn);
}
