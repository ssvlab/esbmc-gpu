/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2004, 2005, 2007, 2008, 2009, 2010 Free Software
   Foundation, Inc.

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

#include "imap4d.h"
#include <gsasl.h>
#include <mailutils/gsasl.h>
#ifdef USE_SQL
# include <mailutils/sql.h>
#endif

static Gsasl *ctx;   
static Gsasl_session *sess_ctx; 

static void auth_gsasl_capa_init (int disable);

static int
create_gsasl_stream (mu_stream_t *newstr, mu_stream_t transport, int flags)
{
  int rc;
  
  rc = mu_gsasl_stream_create (newstr, transport, sess_ctx, flags);
  if (rc)
    {
      mu_diag_output (MU_DIAG_ERROR, _("cannot create SASL stream: %s"),
	      mu_strerror (rc));
      return RESP_NO;
    }

  if ((rc = mu_stream_open (*newstr)) != 0)
    {
      const char *p;
      if (mu_stream_strerror (*newstr, &p))
	p = mu_strerror (rc);
      mu_diag_output (MU_DIAG_ERROR, _("cannot open SASL input stream: %s"), p);
      return RESP_NO;
    }

  return RESP_OK;
}

int
gsasl_replace_streams (void *self, void *data)
{
  mu_stream_t *s = data;

  util_set_input (s[0]);
  util_set_output (s[1]);
  free (s);
  util_event_remove (self);
  free (self);
  return 0;
}

static void
finish_session (void)
{
  gsasl_finish (sess_ctx);
}

static int
auth_gsasl (struct imap4d_command *command, char *auth_type, char **username)
{
  char *input_str = NULL;
  size_t input_size = 0;
  size_t input_len;
  char *output;
  int rc;
  
  rc = gsasl_server_start (ctx, auth_type, &sess_ctx);
  if (rc != GSASL_OK)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("SASL gsasl_server_start: %s"),
 	              gsasl_strerror (rc));
      return 0;
    }

  gsasl_callback_hook_set (ctx, username);

  output = NULL;
  while ((rc = gsasl_step64 (sess_ctx, input_str, &output))
	   == GSASL_NEEDS_MORE)
    {
      util_send ("+ %s\r\n", output);
      imap4d_getline (&input_str, &input_size, &input_len);
    }
  
  if (rc != GSASL_OK)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("GSASL error: %s"),
		      gsasl_strerror (rc));
      free (input_str);
      free (output);
      return RESP_NO;
    }

  /* Some SASL mechanisms output additional data when GSASL_OK is
     returned, and clients must respond with an empty response. */
  if (output[0])
    {
      util_send ("+ %s\r\n", output);
      imap4d_getline (&input_str, &input_size, &input_len);
      if (input_len != 0)
	{
	  mu_diag_output (MU_DIAG_NOTICE, _("non-empty client response"));
          free (input_str);
          free (output);
	  return RESP_NO;
	}
    }

  free (input_str);
  free (output);

  if (*username == NULL)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("GSASL %s: cannot get username"), auth_type);
      return RESP_NO;
    }

  if (sess_ctx)
    {
      mu_stream_t tmp, new_in, new_out;
      mu_stream_t *s;

      util_get_input (&tmp);
      if (create_gsasl_stream (&new_in, tmp, MU_STREAM_READ))
	return RESP_NO;
      util_get_output (&tmp);
      if (create_gsasl_stream (&new_out, tmp, MU_STREAM_WRITE))
	{
	  mu_stream_destroy (&new_in, mu_stream_get_owner (new_in));
	  return RESP_NO;
	}

      s = calloc (2, sizeof (mu_stream_t));
      s[0] = new_in;
      s[1] = new_out;
      util_register_event (STATE_NONAUTH, STATE_AUTH,
			   gsasl_replace_streams, s);
      util_atexit (finish_session);
    }
  
  auth_gsasl_capa_init (1);
  return RESP_OK;
}

static void
auth_gsasl_capa_init (int disable)
{
  int rc;
  char *listmech, *name, *s;

  rc =  gsasl_server_mechlist (ctx, &listmech);
  if (rc != GSASL_OK)
    return;

  for (name = strtok_r (listmech, " ", &s); name;
       name = strtok_r (NULL, " ", &s))
    {
      if (disable)
	auth_remove (name);
      else
	auth_add (strdup (name), auth_gsasl);
    }
      
  free (listmech);
}

#define IMAP_GSSAPI_SERVICE "imap"

static int
retrieve_password (Gsasl *ctx, Gsasl_session *sctx)
{
  char **username = gsasl_callback_hook_get (ctx);
  const char *authid = gsasl_property_get (sctx, GSASL_AUTHID);
  
  if (username && *username == 0)
    *username = strdup (authid);

  if (mu_gsasl_module_data.cram_md5_pwd
      && access (mu_gsasl_module_data.cram_md5_pwd, R_OK) == 0)
    {
      char *key;
      int rc = gsasl_simple_getpass (mu_gsasl_module_data.cram_md5_pwd,
				     authid, &key);
      if (rc == GSASL_OK)
	{
	  gsasl_property_set (sctx, GSASL_PASSWORD, key);
	  free (key);
	  return rc;
	}
    }
  
#ifdef USE_SQL
  if (mu_sql_module_config.password_type == password_plaintext)
    {
      char *passwd;
      int status = mu_sql_getpass (*username, &passwd);
      if (status == 0)
	{
	  gsasl_property_set (sctx, GSASL_PASSWORD, passwd);
	  free (passwd);
	  return GSASL_OK;
	}
    }
#endif
  
  return GSASL_AUTHENTICATION_ERROR; 
}

static int
cb_validate (Gsasl *ctx, Gsasl_session *sctx)
{
  int rc;
  struct mu_auth_data *auth;
  char **username = gsasl_callback_hook_get (ctx);
  const char *authid = gsasl_property_get (sctx, GSASL_AUTHID);
  const char *pass = gsasl_property_get (sctx, GSASL_PASSWORD);

  if (!authid)
    return GSASL_NO_AUTHID;
  if (!pass)
    return GSASL_NO_PASSWORD;
  
  *username = strdup (authid);
  
  auth = mu_get_auth_by_name (*username);

  if (auth == NULL)
    return GSASL_AUTHENTICATION_ERROR;

  rc = mu_authenticate (auth, pass);
  mu_auth_data_free (auth);
  
  return rc == 0 ? GSASL_OK : GSASL_AUTHENTICATION_ERROR;
}
  
static int
callback (Gsasl *ctx, Gsasl_session *sctx, Gsasl_property prop)
{
    int rc = GSASL_OK;

    switch (prop) {
    case GSASL_PASSWORD:
      rc = retrieve_password (ctx, sctx);
      break;
      
    case GSASL_SERVICE:
      gsasl_property_set (sctx, prop,
			  mu_gsasl_module_data.service ?
			    mu_gsasl_module_data.service :IMAP_GSSAPI_SERVICE);
      break;
      
    case GSASL_REALM:
      gsasl_property_set (sctx, prop,
			  mu_gsasl_module_data.realm ?
			    mu_gsasl_module_data.realm : util_localname ());
      break;
      
    case GSASL_HOSTNAME:
      gsasl_property_set (sctx, prop,
			  mu_gsasl_module_data.hostname ?
			    mu_gsasl_module_data.hostname : util_localname ());
      break;
      
#if 0
    FIXME:
    case GSASL_VALIDATE_EXTERNAL:
    case GSASL_VALIDATE_SECURID:
#endif
      
    case GSASL_VALIDATE_SIMPLE:
      rc = cb_validate (ctx, sctx);
      break;
      
    case GSASL_VALIDATE_ANONYMOUS:
      if (mu_gsasl_module_data.anon_user)
	{
	  char **username = gsasl_callback_hook_get (ctx);
	  mu_diag_output (MU_DIAG_INFO, _("anonymous user %s logged in"),
			  gsasl_property_get (sctx, GSASL_ANONYMOUS_TOKEN));
	  *username = strdup (mu_gsasl_module_data.anon_user);
	}
      else
	{
	  mu_diag_output (MU_DIAG_ERR,
			  _("attempt to log in as anonymous user denied"));
	}
      break;
      
    case GSASL_VALIDATE_GSSAPI:
      {
	char **username = gsasl_callback_hook_get (ctx);
	*username = strdup (gsasl_property_get(sctx, GSASL_AUTHZID));
	break;
      }
      
    default:
	rc = GSASL_NO_CALLBACK;
	mu_error (_("unsupported callback property %d"), prop);
	break;
    }

    return rc;
}

void
auth_gsasl_init ()
{
  int rc;

  rc = gsasl_init (&ctx);
  if (rc != GSASL_OK)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("cannot initialize libgsasl: %s"),
	      gsasl_strerror (rc));
    }

  gsasl_callback_set (ctx, callback);
  
  auth_gsasl_capa_init (0);
}

