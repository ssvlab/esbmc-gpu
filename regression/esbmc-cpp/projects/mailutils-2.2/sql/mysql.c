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
#include <mailutils/cctype.h>
#include <mailutils/mailutils.h>
#include <mailutils/sql.h>

#include <mysql/mysql.h>
#include <mysql/errmsg.h>
#include <mailutils/sha1.h>

struct mu_mysql_data
{
  MYSQL *mysql;
  MYSQL_RES *result;
};
  

static int 
do_mysql_query (mu_sql_connection_t conn, char *query)
{
  int rc;
  int i;
  MYSQL *mysql;

  for (i = 0; i < 10; i++)
    {
      mysql = ((struct mu_mysql_data*)conn->data)->mysql;
      rc = mysql_query (mysql, query);
      if (rc && mysql_errno (mysql) == CR_SERVER_GONE_ERROR)
	{
	  /* Reconnect? */
	  mu_sql_disconnect (conn);
	  mu_sql_connect (conn);
	  continue;
	}
      break;
    }
  return rc;
}

/* ************************************************************************* */
/* Interface routines */

static int
mu_mysql_init (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = calloc (1, sizeof (*mp));
  if (!mp)
    return ENOMEM;
  conn->data = mp;
  return 0;
}

static int
mu_mysql_destroy (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  free (mp->mysql);
  free (mp);
  conn->data = NULL;
  return 0;
}
  

static int
mu_mysql_connect (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  char *host, *socket_name = NULL;
  
  mp->mysql = malloc (sizeof(MYSQL));
  if (!mp->mysql)
    return ENOMEM;
  
  mysql_init (mp->mysql);

  if (conn->server && conn->server[0] == '/')
    {
      host = "localhost";
      socket_name = conn->server;
    }
  else
    host = conn->server;
  
  if (!mysql_real_connect(mp->mysql, 
			  host,
			  conn->login,
			  conn->password,
			  conn->dbname,
			  conn->port,
			  socket_name,
			  CLIENT_MULTI_RESULTS))
    return MU_ERR_SQL;
  
  return 0;
}

static int 
mu_mysql_disconnect (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  
  mysql_close (mp->mysql);
  free (mp->mysql);
  mp->mysql = NULL;  
  return 0;
}

static int
mu_mysql_query (mu_sql_connection_t conn, char *query)
{
  if (do_mysql_query (conn, query)) 
    return MU_ERR_SQL;
  return 0;
}


static int
mu_mysql_store_result (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  if (!(mp->result = mysql_store_result (mp->mysql)))
    {
      if (mysql_errno (mp->mysql))
	return MU_ERR_SQL;
      return MU_ERR_NO_RESULT;
    }
  return 0;
}

static int
mu_mysql_release_result (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  mysql_free_result (mp->result);
  mp->result = NULL;
  return 0;
}

static int
mu_mysql_num_columns (mu_sql_connection_t conn, size_t *np)
{
  struct mu_mysql_data *mp = conn->data;
  *np = mysql_num_fields (mp->result);
  return 0;
}

static int
mu_mysql_num_tuples (mu_sql_connection_t conn, size_t *np)
{
  struct mu_mysql_data *mp = conn->data;
  *np = mysql_num_rows (mp->result);
  return 0;
}

static int
mu_mysql_get_column (mu_sql_connection_t conn, size_t nrow, size_t ncol,
		     char **pdata)
{
  struct mu_mysql_data *mp = conn->data;
  MYSQL_ROW row;

  if (nrow >= mysql_num_rows (mp->result)
      || ncol >= mysql_num_fields (mp->result))
    return MU_ERR_BAD_COLUMN;
  
  mysql_data_seek (mp->result, nrow);
  row = mysql_fetch_row (mp->result);
  if (!row)
    return MU_ERR_BAD_COLUMN;
  *pdata = row[ncol];
  return 0;
}

static int
mu_mysql_get_field_number (mu_sql_connection_t conn, const char *fname,
			   size_t *fno)
{
  struct mu_mysql_data *mp = conn->data;
  MYSQL_FIELD *fields;
  size_t nf, i;
  
  if (!mp->result)
    return MU_ERR_NO_RESULT;

  fields = mysql_fetch_fields (mp->result);
  nf = mysql_num_fields (mp->result);
  for (i = 0; i < nf; i++)
    if (strcmp (fname, fields[i].name) == 0)
      {
	*fno = i;
	return 0;
      }
  return MU_ERR_NOENT;
}

static const char *
mu_mysql_errstr (mu_sql_connection_t conn)
{
  struct mu_mysql_data *mp = conn->data;
  return mysql_error (mp->mysql);
}


/* MySQL scrambled password support */

/* Convert a single hex digit to corresponding number */
static unsigned 
digit_to_number (char c)
{
  return (unsigned) (c >= '0' && c <= '9' ? c-'0' :
                     c >= 'A' && c <= 'Z' ? c-'A'+10 :
                     c-'a'+10);
}

/* Extract salt value from MySQL scrambled password.
   
   WARNING: The code assumes that
       1. strlen (password) % 8 == 0
       2. number_of_entries (RES) = strlen (password) / 8

   For MySQL >= 3.21, strlen(password) == 16 */
static void
get_salt_from_scrambled (unsigned long *res, const char *password)
{
  res[0] = res[1] = 0;
  while (*password)
    {
      unsigned long val = 0;
      unsigned i;

      for (i = 0; i < 8 ; i++)
        val = (val << 4) + digit_to_number (*password++);
      *res++ = val;
    }
}

/* Scramble a plaintext password */
static void
scramble_password (unsigned long *result, const char *password)
{
  unsigned long nr = 1345345333L, add = 7, nr2 = 0x12345671L;
  unsigned long tmp;

  for (; *password ; password++)
    {
      if (*password == ' ' || *password == '\t')
        continue;                   
      tmp = (unsigned long) (unsigned char) *password;
      nr ^= (((nr & 63) + add) * tmp)+ (nr << 8);
      nr2 += (nr2 << 8) ^ nr;
      add += tmp;
    }

  result[0] = nr & (((unsigned long) 1L << 31) -1L);
  result[1] = nr2 & (((unsigned long) 1L << 31) -1L);
}

static void
mu_octet_to_hex (char *to, const unsigned char *str, unsigned len)
{
  const unsigned char *str_end= str + len;
  static char d[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  for ( ; str != str_end; ++str)
    {
      *to++ = d[(*str & 0xF0) >> 4];
      *to++ = d[*str & 0x0F];
    }
  *to= '\0';
}

#define SHA1_HASH_SIZE 20
static int
mu_check_mysql_4x_password (const char *scrambled, const char *message)
{
  struct mu_sha1_ctx sha1_context;
  unsigned char hash_stage2[SHA1_HASH_SIZE];
  char to[2*SHA1_HASH_SIZE + 2];

  /* stage 1: hash password */
  mu_sha1_init_ctx (&sha1_context);
  mu_sha1_process_bytes (message, strlen (message), &sha1_context);
  mu_sha1_finish_ctx (&sha1_context, to);

  /* stage 2: hash stage1 output */
  mu_sha1_init_ctx (&sha1_context);
  mu_sha1_process_bytes (to, SHA1_HASH_SIZE, &sha1_context);
  mu_sha1_finish_ctx (&sha1_context, hash_stage2);

  /* convert hash_stage2 to hex string */
  to[0] = '*';
  mu_octet_to_hex (to + 1, hash_stage2, SHA1_HASH_SIZE);

  /* Compare both strings */
  return memcmp (to, scrambled, strlen (scrambled));
}

static int
mu_check_mysql_3x_password (const char *scrambled, const char *message)
{
  unsigned long hash_pass[2], hash_message[2];
  char buf[17];

  memcpy (buf, scrambled, 16);
  buf[16] = 0;
  scrambled = buf;
  
  get_salt_from_scrambled (hash_pass, scrambled);
  scramble_password (hash_message, message);
  return !(hash_message[0] == hash_pass[0]
	   && hash_message[1] == hash_pass[1]);
}

/* Check whether a plaintext password MESSAGE matches MySQL scrambled password
   PASSWORD */
int
mu_check_mysql_scrambled_password (const char *scrambled, const char *message)
{
  const char *p;

  /* Try to normalize it by cutting off trailing whitespace */
  for (p = scrambled + strlen (scrambled) - 1;
       p > scrambled && mu_isspace (*p); p--)
    ;
  switch (p - scrambled)
    {
    case 15:
      return mu_check_mysql_3x_password (scrambled, message);
    case 40:
      return mu_check_mysql_4x_password (scrambled, message);
    }
  return 1;
}


/* Register module */
MU_DECL_SQL_DISPATCH_T(mysql) = {
  "mysql",
  3306,
  mu_mysql_init,
  mu_mysql_destroy,
  mu_mysql_connect,
  mu_mysql_disconnect,
  mu_mysql_query,
  mu_mysql_store_result,
  mu_mysql_release_result,
  mu_mysql_num_tuples,
  mu_mysql_num_columns,
  mu_mysql_get_column,
  mu_mysql_get_field_number,
  mu_mysql_errstr,
};

