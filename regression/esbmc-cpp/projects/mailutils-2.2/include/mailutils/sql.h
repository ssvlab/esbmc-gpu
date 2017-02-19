/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2009, 2010 Free Software Foundation,
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

#ifndef _MAILUTILS_SQL_H
#define _MAILUTILS_SQL_H

/* Configuration */
enum mu_password_type
  {
    password_plaintext,       /* Plaintext passwords */
    password_scrambled,       /* Scrambled MySQL (>=3.21) password */
    password_hash,            /* MD5 (or DES or whatever) hash */
  };

struct mu_sql_module_config
{
  char *interface;
  char *getpwnam_query;
  char *getpass_query;
  char *getpwuid_query;
  char *host;
  char *user;
  char *passwd;
  char *db;
  int port;
  enum mu_password_type password_type;
  int positional;
  mu_assoc_t field_map;
};

/* FIXME: Should not be here, but needed for several other sources
   (imap4d/auth_gsasl.c, for instance) */
struct mu_internal_sql_config
{
  int interface;
  char *getpwnam_query;
  char *getpass_query;
  char *getpwuid_query;
  char *host;
  char *user;
  char *passwd;
  char *db;
  int port;
  enum mu_password_type password_type;
  int positional;
  mu_assoc_t field_map;
};

extern struct mu_internal_sql_config mu_sql_module_config;

/* Loadable Modules Support */
#define __s_cat2__(a,b) a ## b
#define __s_cat3__(a,b,c) a ## b ## c
#define RDL_EXPORT(module,name) __s_cat3__(module,_LTX_,name)

typedef int (*mu_rdl_init_t) (void);
typedef void (*mu_rdl_done_t) (void);

#ifdef _HAVE_LIBLTDL /*FIXME: Remove leading _ when SQL + ltdl works*/
# define MU_DECL_SQL_DISPATCH_T(mod) \
  mu_sql_dispatch_t RDL_EXPORT(mod,dispatch_tab)
#else
# define MU_DECL_SQL_DISPATCH_T(mod) \
  mu_sql_dispatch_t __s_cat2__(mod,_dispatch_tab)
#endif

enum mu_sql_connection_state
  {
    mu_sql_not_connected,
    mu_sql_connected,
    mu_sql_query_run,
    mu_sql_result_available
  };

typedef struct mu_sql_connection *mu_sql_connection_t;

struct mu_sql_connection
{
  int  interface;
  char *server;
  int  port;
  char *login;
  char *password;
  char *dbname;
  void *data;
  enum mu_sql_connection_state state;
};

typedef struct mu_sql_dispatch mu_sql_dispatch_t;

struct mu_sql_dispatch
{
  char *name;
  int port;

  int (*init) (mu_sql_connection_t conn);
  int (*destroy) (mu_sql_connection_t conn);

  int (*connect) (mu_sql_connection_t conn);
  int (*disconnect) (mu_sql_connection_t conn);

  int (*query) (mu_sql_connection_t conn, char *query);
  int (*store_result) (mu_sql_connection_t conn);
  int (*release_result) (mu_sql_connection_t conn);

  int (*num_tuples) (mu_sql_connection_t conn, size_t *np);
  int (*num_columns) (mu_sql_connection_t conn, size_t *np);

  int (*get_column) (mu_sql_connection_t conn, size_t nrow, size_t ncol,
		     char **pdata);

  int (*get_field_number) (mu_sql_connection_t conn, const char *fname,
			   size_t *fno);

  const char *(*errstr) (mu_sql_connection_t conn);
};

/* Public interfaces */
int mu_sql_interface_index (char *name);

int mu_sql_connection_init (mu_sql_connection_t *conn, int interface,
			    char *server, int  port, char *login,
			    char *password, char *dbname);
int mu_sql_connection_destroy (mu_sql_connection_t *conn);

int mu_sql_connect (mu_sql_connection_t conn);
int mu_sql_disconnect (mu_sql_connection_t conn);

int mu_sql_query (mu_sql_connection_t conn, char *query);

int mu_sql_store_result (mu_sql_connection_t conn);
int mu_sql_release_result (mu_sql_connection_t conn);

int mu_sql_num_tuples (mu_sql_connection_t conn, size_t *np);
int mu_sql_num_columns (mu_sql_connection_t conn, size_t *np);

int mu_sql_get_column (mu_sql_connection_t conn, size_t nrow, size_t ncol,
		       char **pdata);
int mu_sql_get_field (mu_sql_connection_t conn, size_t nrow, const char *fname,
		      char **pdata);

const char *mu_sql_strerror (mu_sql_connection_t conn);

extern char *mu_sql_expand_query (const char *query, const char *ustr);
extern int mu_sql_getpass (const char *username, char **passwd);
extern int mu_check_mysql_scrambled_password (const char *scrambled,
					      const char *message);

int mu_sql_decode_password_type (const char *arg, enum mu_password_type *t);

#endif
