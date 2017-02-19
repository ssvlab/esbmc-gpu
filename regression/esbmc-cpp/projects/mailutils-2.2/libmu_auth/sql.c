/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

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
#else
# include <string.h>
#endif
#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif

#include <mailutils/assoc.h>
#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <mailutils/mailbox.h>
#include <mailutils/mu_auth.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <mailutils/mutil.h>
#include <mailutils/sql.h>
#include <mailutils/vartab.h>
#include <mailutils/cstr.h>
#include "sql.h"

#ifdef USE_SQL

struct mu_internal_sql_config mu_sql_module_config;

static char *
sql_escape_string (const char *ustr)
{
  char *str, *q;
  const unsigned char *p;
  size_t len = strlen (ustr);
#define ESCAPABLE_CHAR "\\'\""
  
  for (p = (const unsigned char *) ustr; *p; p++)
    {
      if (strchr (ESCAPABLE_CHAR, *p))
	len++;
    }

  str = malloc (len + 1);
  if (!str)
    return NULL;

  for (p = (const unsigned char *) ustr, q = str; *p; p++)
    {
      if (strchr (ESCAPABLE_CHAR, *p))
	*q++ = '\\';
      *q++ = *p;
    }
  *q = 0;
  return str;
}

char *
mu_sql_expand_query (const char *query, const char *ustr)
{
  int rc;
  char *res;
  char *esc_ustr;
  mu_vartab_t vtab;
  
  if (!query)
    return NULL;

  esc_ustr = sql_escape_string (ustr);
  mu_vartab_create (&vtab);
  mu_vartab_define (vtab, "user", ustr, 1);
  mu_vartab_define (vtab, "u", ustr, 1);
  rc = mu_vartab_expand (vtab, query, &res);
  if (rc)
    res = NULL;
  mu_vartab_destroy (&vtab);

  free (esc_ustr);
  return res;
}


static int
decode_tuple_v1_0 (mu_sql_connection_t conn, int n,
		   struct mu_auth_data **return_data)
{
  int rc;
  char *mailbox_name = NULL;
  char *name;
      
  if (mu_sql_get_column (conn, 0, 0, &name))
    return MU_ERR_FAILURE;

  if (n == 7)
    {
      char *tmp;
      if (mu_sql_get_column (conn, 0, 6, &tmp))
	return MU_ERR_FAILURE;
      if (tmp && (mailbox_name = strdup (tmp)) == NULL)
	return ENOMEM;
    }
  else if (mu_construct_user_mailbox_url (&mailbox_name, name))
    return MU_ERR_FAILURE;
      
  if (mailbox_name)
    {
      char *passwd, *suid, *sgid, *dir, *shell;
	  
      if (mu_sql_get_column (conn, 0, 1, &passwd)
	  || !passwd
	  || mu_sql_get_column (conn, 0, 2, &suid)
	  || !suid
	  || mu_sql_get_column (conn, 0, 3, &sgid)
	  || !sgid
	  || mu_sql_get_column (conn, 0, 4, &dir)
	  || !dir
	  || mu_sql_get_column (conn, 0, 5, &shell)
	  || !shell)
	return MU_ERR_FAILURE;
      
      rc = mu_auth_data_alloc (return_data,
			       name,
			       passwd,
			       atoi (suid),
			       atoi (sgid),
			       "SQL User",
			       dir,
			       shell,
			       mailbox_name,
			       1);
    }
  else
    rc = MU_ERR_AUTH_FAILURE;
  
  free (mailbox_name);
  return rc;
}

static int
get_field (mu_sql_connection_t conn, const char *id, char **ret, int mandatory)
{
  const char **name = mu_assoc_ref (mu_sql_module_config.field_map, id);
  int rc = mu_sql_get_field (conn, 0, name ? *name : id, ret);
  if (rc)
    {
      if (mandatory || rc != MU_ERR_NOENT)
	mu_error (_("cannot get SQL field `%s' (`%s'): %s"),
		  id, name ? *name : id, mu_strerror (rc));
    }
  else if (!*ret)
    {
      if (mandatory)
	{
	  mu_error (_("SQL field `%s' (`%s') has NULL value"),
		    id, name ? *name : id);
	  rc = MU_ERR_READ;
	}
      else
	rc = MU_ERR_NOENT;
    }

  return rc;
}

static int
decode_tuple_new (mu_sql_connection_t conn, int n,
		  struct mu_auth_data **return_data)
{
  int rc;
  char *mailbox_name = NULL;
  char *name;
  char *passwd, *suid, *sgid, *dir, *shell, *gecos, *squota;
  mu_off_t quota = 0;
  char *p;
  uid_t uid;
  gid_t gid;

  if (get_field (conn, MU_AUTH_NAME, &name, 1)
      || get_field (conn, MU_AUTH_PASSWD, &passwd, 1)
      || get_field (conn, MU_AUTH_UID, &suid, 1)
      || get_field (conn, MU_AUTH_GID, &sgid, 1)     
      || get_field (conn, MU_AUTH_DIR, &dir, 1)     
      || get_field (conn, MU_AUTH_SHELL, &shell, 1))
    return MU_ERR_FAILURE;

  if (get_field (conn, MU_AUTH_GECOS, &gecos, 0))
    gecos = "SQL user";
  
  uid = strtoul (suid, &p, 0);
  if (*p)
    {
      mu_error (_("invalid value for uid: %s"), suid);
      return MU_ERR_FAILURE;
    }

  gid = strtoul (sgid, &p, 0);
  if (*p)
    {
      mu_error (_("invalid value for gid: %s"), sgid);
      return MU_ERR_FAILURE;
    }
  
  rc = get_field (conn, MU_AUTH_MAILBOX, &mailbox_name, 0);
  switch (rc)
    {
    case 0:
      mailbox_name = strdup (mailbox_name);
      break;
      
    case MU_ERR_NOENT:
      if (mu_construct_user_mailbox_url (&mailbox_name, name))
	return MU_ERR_FAILURE;
      break;

    default:
      return MU_ERR_FAILURE;
    }

  rc = get_field (conn, MU_AUTH_QUOTA, &squota, 0);
  if (rc == 0)
    {
      if (mu_c_strcasecmp (squota, "none") == 0)
	quota = 0;
      else
	{
	  quota = strtoul (squota, &p, 10);
	  switch (*p)
	    {
	    case 0:
	      break;
	      
	    case 'k':
	    case 'K':
	      quota *= 1024;
	      break;
      
	    case 'm':
	    case 'M':
	      quota *= 1024*1024;
	      break;
	      
	    default:
	      mu_error (_("invalid value for quota: %s"), squota);
	      free (mailbox_name);
	      return MU_ERR_FAILURE;
	    }
	}
    }
  else if (rc ==  MU_ERR_NOENT)
    quota = 0;
  else
    {
      free (mailbox_name);
      return MU_ERR_FAILURE;
    }

  rc = mu_auth_data_alloc (return_data,
			   name,
			   passwd,
			   uid,
			   gid,
			   gecos,
			   dir,
			   shell,
			   mailbox_name,
			   1);
  if (rc == 0)
    mu_auth_data_set_quota (*return_data, quota);
  
  free (mailbox_name);
  return rc;
}  

static int
decode_tuple (mu_sql_connection_t conn, int n,
	      struct mu_auth_data **return_data)
{
  if (mu_sql_module_config.field_map || !mu_sql_module_config.positional)
    return decode_tuple_new (conn, n, return_data);
  else
    return decode_tuple_v1_0 (conn, n, return_data);
}
 
static int
mu_auth_sql_by_name (struct mu_auth_data **return_data,
		     const void *key,
		     void *func_data MU_ARG_UNUSED,
		     void *call_data MU_ARG_UNUSED)
{
  int status, rc;
  char *query_str = NULL;
  mu_sql_connection_t conn;
  size_t n;
  
  if (!key)
    return EINVAL;

  query_str = mu_sql_expand_query (mu_sql_module_config.getpwnam_query, key);

  if (!query_str)
    return MU_ERR_FAILURE;

  status = mu_sql_connection_init (&conn,
				   mu_sql_module_config.interface,
				   mu_sql_module_config.host,
				   mu_sql_module_config.port,
				   mu_sql_module_config.user,
				   mu_sql_module_config.passwd,
				   mu_sql_module_config.db);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_connect (conn);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return EAGAIN;
    }
  
  status = mu_sql_query (conn, query_str);
  free (query_str);
  
  if (status)
    {
      mu_error (_("SQL query failed: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_store_result (conn);

  if (status)
    {
      mu_error (_("cannot store SQL result: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  mu_sql_num_tuples (conn, &n);
  if (n == 0)
    rc = MU_ERR_AUTH_FAILURE;
  else
    rc = decode_tuple (conn, n, return_data);
  
  mu_sql_release_result (conn);
  mu_sql_disconnect (conn);
  mu_sql_connection_destroy (&conn);
  
  return rc;
}

static int
mu_auth_sql_by_uid (struct mu_auth_data **return_data,
		    const void *key,
		    void *func_data MU_ARG_UNUSED,
		    void *call_data MU_ARG_UNUSED)
{
  char uidstr[64];
  int status, rc;
  char *query_str = NULL;
  mu_sql_connection_t conn;
  size_t n;
  
  if (!key)
    return EINVAL;

  snprintf (uidstr, sizeof (uidstr), "%u", *(uid_t*)key);
  query_str = mu_sql_expand_query (mu_sql_module_config.getpwuid_query,
				   uidstr);

  if (!query_str)
    return ENOMEM;

  status = mu_sql_connection_init (&conn,
				   mu_sql_module_config.interface,
				   mu_sql_module_config.host,
				   mu_sql_module_config.port,
				   mu_sql_module_config.user,
				   mu_sql_module_config.passwd,
				   mu_sql_module_config.db);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_connect (conn);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return EAGAIN;
    }
  
  status = mu_sql_query (conn, query_str);
  free (query_str);
  
  if (status)
    {
      mu_error (_("SQL query failed: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_store_result (conn);

  if (status)
    {
      mu_error (_("cannot store SQL result: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  mu_sql_num_tuples (conn, &n);

  if (n == 0)
    rc = MU_ERR_AUTH_FAILURE;
  else
    rc = decode_tuple (conn, n, return_data);
  
  mu_sql_release_result (conn);
  mu_sql_disconnect (conn);
  mu_sql_connection_destroy (&conn);
  
  return rc;
}

int
mu_sql_getpass (const char *username, char **passwd)
{
  mu_sql_connection_t conn;
  char *query_str;
  int status;
  char *sql_pass;
  
  query_str = mu_sql_expand_query (mu_sql_module_config.getpass_query, username);

  if (!query_str)
    return MU_ERR_FAILURE;

  status = mu_sql_connection_init (&conn,
				   mu_sql_module_config.interface,
				   mu_sql_module_config.host,
				   mu_sql_module_config.port,
				   mu_sql_module_config.user,
				   mu_sql_module_config.passwd,
				   mu_sql_module_config.db);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_connect (conn);

  if (status)
    {
      mu_error ("%s: %s", mu_strerror (status), mu_sql_strerror (conn));
      mu_sql_connection_destroy (&conn);
      free (query_str);
      return EAGAIN;
    }
  
  status = mu_sql_query (conn, query_str);
  free (query_str);
  
  if (status)
    {
      mu_error (_("SQL query failed: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_store_result (conn);

  if (status)
    {
      mu_error (_("cannot store SQL result: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  status = mu_sql_get_column (conn, 0, 0, &sql_pass);
  if (status)
    {
      mu_error (_("cannot get password from SQL: %s"),
		(status == MU_ERR_SQL) ?  mu_sql_strerror (conn) :
	 	                          mu_strerror (status));
      mu_sql_release_result (conn);
      mu_sql_connection_destroy (&conn);
      return MU_ERR_FAILURE;
    }

  *passwd = strdup (sql_pass);

  mu_sql_disconnect (conn);
  mu_sql_connection_destroy (&conn);

  if (!*passwd)
    return ENOMEM;

  return 0;
}

static int
mu_sql_authenticate (struct mu_auth_data **return_data MU_ARG_UNUSED,
		     const void *key,
		     void *func_data MU_ARG_UNUSED, void *call_data)
{
  const struct mu_auth_data *auth_data = key;
  char *pass = call_data;
  char *sql_pass;
  int rc;
  
  if (!auth_data)
    return EINVAL;

  if ((rc = mu_sql_getpass (auth_data->name, &sql_pass)))
    return rc;

  switch (mu_sql_module_config.password_type)
    {
    case password_hash:
      rc = strcmp (sql_pass, crypt (pass, sql_pass));
      break;

    case password_scrambled:
      /* FIXME: Should this call be implementation-independent? I mean,
         should we have mu_sql_check_scrambled() that will match the
	 password depending on the exact type of the underlying database,
	 just as the rest of mu_sql_.* functions do */
#ifdef HAVE_MYSQL
      rc = mu_check_mysql_scrambled_password (sql_pass, pass);
#else
      rc = 1;
#endif
      break;

    case password_plaintext:
      rc = strcmp (sql_pass, pass);
      break;
    }

  free (sql_pass);
  
  return rc == 0 ? 0 : MU_ERR_AUTH_FAILURE;
}

int
mu_sql_module_init (enum mu_gocs_op op, void *data)
{
  struct mu_sql_module_config *cfg = data;

  if (op != mu_gocs_op_set)
    return 0;
  mu_sql_module_config.interface = mu_sql_interface_index (cfg->interface);
  if (mu_sql_module_config.interface == 0)
    {
      mu_error (_("unknown SQL interface `%s'"), cfg->interface);
      return 1;
    }

  mu_sql_module_config.getpwnam_query = cfg->getpwnam_query;   
  mu_sql_module_config.getpass_query  = cfg->getpass_query;    
  mu_sql_module_config.getpwuid_query = cfg->getpwuid_query;   
  mu_sql_module_config.host = cfg->host;             
  mu_sql_module_config.user = cfg->user;             
  mu_sql_module_config.passwd = cfg->passwd;           
  mu_sql_module_config.db = cfg->db;               
  mu_sql_module_config.port = cfg->port;             
  mu_sql_module_config.password_type = cfg->password_type;    
  mu_sql_module_config.field_map = cfg->field_map;        

  return 0;
}

#else

# define mu_sql_authenticate mu_auth_nosupport
# define mu_auth_sql_by_name mu_auth_nosupport
# define mu_auth_sql_by_uid mu_auth_nosupport

#endif


struct mu_auth_module mu_auth_sql_module = {
  "sql",
#ifdef USE_SQL
  mu_sql_module_init,
#else
  NULL,
#endif
  mu_sql_authenticate,
  NULL,
  mu_auth_sql_by_name,
  NULL,
  mu_auth_sql_by_uid,
  NULL
};

