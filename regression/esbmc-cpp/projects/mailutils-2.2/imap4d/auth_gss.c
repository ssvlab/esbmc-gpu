/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2004, 2007, 2009, 2010 Free
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

/*
  GSSAPI authentication for imap (rfc 1731). 
 */

#include "imap4d.h"

#include <netinet/in.h>

#ifdef WITH_GSS
# include <gss.h>
#else
# include <krb5.h>
# ifdef HAVE_GSSAPI_H
#  include <gssapi.h>
# else
#  ifdef HAVE_GSSAPI_GSSAPI_H
#   include <gssapi/gssapi.h>
#  endif
#  ifdef HAVE_GSSAPI_GSSAPI_GENERIC_H
#   include <gssapi/gssapi_generic.h>
#  endif
# endif
#endif

#define GSS_AUTH_P_NONE      1
#define GSS_AUTH_P_INTEGRITY 2
#define GSS_AUTH_P_PRIVACY   4

#define SUPPORTED_P_MECH GSS_AUTH_P_NONE
static int protection_mech;
size_t server_buffer_size = 8192;
size_t client_buffer_size;

static void
display_status_1 (char *m, OM_uint32 code, int type)
{
  OM_uint32 maj_stat, min_stat;
  gss_buffer_desc msg;
  OM_uint32 msg_ctx;

  msg_ctx = 0;
  do
    {
      maj_stat = gss_display_status (&min_stat, code,
				     type, GSS_C_NO_OID, &msg_ctx, &msg);
      if (GSS_ERROR (maj_stat))
	{
	  asprintf ((char**)&msg.value, "code %d", code);
	  msg.length = strlen (msg.value);
	}

      mu_diag_output (MU_DIAG_ERROR, _("GSS-API error %s (%s): %.*s"),
	      m, type == GSS_C_GSS_CODE ? _("major") : _("minor"),
	      (int) msg.length, (char *) msg.value);

      if (GSS_ERROR (maj_stat))
	free (msg.value);
      else
        gss_release_buffer (&min_stat, &msg);
    }
  while (!GSS_ERROR (maj_stat) && msg_ctx);
}

static void
display_status (char *msg, OM_uint32 maj_stat, OM_uint32 min_stat)
{
  display_status_1 (msg, maj_stat, GSS_C_GSS_CODE);
  display_status_1 (msg, min_stat, GSS_C_MECH_CODE);
}

#ifndef WITH_GSS
static int
imap4d_gss_userok (gss_buffer_t client_name, char *name)
{
  int rc = -1;
  krb5_principal p;
  krb5_context kcontext;

  krb5_init_context (&kcontext);

  if (krb5_parse_name (kcontext, client_name->value, &p) != 0)
    return -1;
  if (krb5_kuserok (kcontext, p, name))
    rc = 0;
  else
    rc = 1;
  krb5_free_principal (kcontext, p);
  return rc;
}
#endif

static int
auth_gssapi (struct imap4d_command *command,
	     char *auth_type_unused, char **username)
{
  gss_buffer_desc tokbuf, outbuf;
  OM_uint32 maj_stat, min_stat, min_stat2;
  int cflags;
  OM_uint32 sec_level, mech;
  gss_ctx_id_t context;
  gss_cred_id_t cred_handle, server_creds;
  gss_OID mech_type;
  char *token_str = NULL;
  size_t token_size = 0;
  size_t token_len;
  unsigned char *tmp = NULL;
  size_t size;
  gss_name_t server_name;
  gss_qop_t quality;
  gss_name_t client;
  gss_buffer_desc client_name;
  int baduser;

  /* Obtain server credentials. RFC 1732 states, that 
     "The server must issue a ready response with no data and pass the
     resulting client supplied token to GSS_Accept_sec_context as
     input_token, setting acceptor_cred_handle to NULL (for "use default
     credentials"), and 0 for input_context_handle (initially)."
     In MIT implementation, passing NULL as acceptor_cred_handle won't
     work (possibly due to a bug in krb5_gss_accept_sec_context()), so
     we acquire server credentials explicitly. */

  asprintf ((char **) &tmp, "imap@%s", util_localname ());
  tokbuf.value = tmp;
  tokbuf.length = strlen (tokbuf.value) + 1;
  maj_stat = gss_import_name (&min_stat, &tokbuf,
			      GSS_C_NT_HOSTBASED_SERVICE, &server_name);
  if (maj_stat != GSS_S_COMPLETE)
    {
      display_status ("import name", maj_stat, min_stat);
      return RESP_NO;
    }

  maj_stat = gss_acquire_cred (&min_stat, server_name, 0,
			       GSS_C_NULL_OID_SET, GSS_C_ACCEPT,
			       &server_creds, NULL, NULL);
  gss_release_name (&min_stat2, &server_name);

  if (maj_stat != GSS_S_COMPLETE)
    {
      display_status ("acquire credentials", maj_stat, min_stat);
      return RESP_NO;
    }

  /* Start the dialogue */

  util_send ("+ \r\n");
  util_flush_output ();
  
  context = GSS_C_NO_CONTEXT;

  for (;;)
    {
      OM_uint32 ret_flags;
      
      imap4d_getline (&token_str, &token_size, &token_len);
      mu_base64_decode ((unsigned char*) token_str, token_len, &tmp, &size);
      tokbuf.value = tmp;
      tokbuf.length = size;

      maj_stat = gss_accept_sec_context (&min_stat,
					 &context,
					 server_creds,
					 &tokbuf,
					 GSS_C_NO_CHANNEL_BINDINGS,
					 &client,
					 &mech_type,
					 &outbuf,
					 &ret_flags, NULL, &cred_handle);
      free (tmp);
      if (maj_stat == GSS_S_CONTINUE_NEEDED)
	{
	  if (outbuf.length)
	    {
	      mu_base64_encode (outbuf.value, outbuf.length, &tmp, &size);
	      util_send ("+ %s\r\n", tmp);
	      free (tmp);
	      gss_release_buffer (&min_stat, &outbuf);
	    }
	  continue;
	}
      else if (maj_stat == GSS_S_COMPLETE)
	break;
      /* Bail out otherwise */

      display_status ("accept context", maj_stat, min_stat);
      maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
      gss_release_buffer (&min_stat, &outbuf);
      free (token_str);
      return RESP_NO;
    }

  if (outbuf.length)
    {
      mu_base64_encode (outbuf.value, outbuf.length, &tmp, &size);
      util_send ("+ %s\r\n", tmp);
      free (tmp);
      gss_release_buffer (&min_stat, &outbuf);
      imap4d_getline (&token_str, &token_size, &token_len);
    }

  /* Construct security-level data */
  sec_level = htonl ((SUPPORTED_P_MECH << 24) | server_buffer_size);
  tokbuf.length = 4;
  tokbuf.value = &sec_level;
  maj_stat = gss_wrap (&min_stat, context, 0, GSS_C_QOP_DEFAULT,
		       &tokbuf, &cflags, &outbuf);
  if (maj_stat != GSS_S_COMPLETE)
    {
      display_status ("wrap", maj_stat, min_stat);
      free (token_str);
      return RESP_NO;
    }
  
  mu_base64_encode (outbuf.value, outbuf.length, &tmp, &size);
  util_send ("+ %s\r\n", tmp);
  free (tmp);

  imap4d_getline (&token_str, &token_size, &token_len);
  mu_base64_decode ((unsigned char *) token_str, token_len,
		    (unsigned char **) &tokbuf.value, &tokbuf.length);
  free (token_str);

  maj_stat = gss_unwrap (&min_stat, context, &tokbuf, &outbuf, &cflags,
			 &quality);
  free (tokbuf.value);
  if (maj_stat != GSS_S_COMPLETE)
    {
      display_status ("unwrap", maj_stat, min_stat);
      return RESP_NO;
    }
  
  sec_level = ntohl (*(OM_uint32 *) outbuf.value);

  /* FIXME: parse sec_level and act accordingly to its settings */
  mech = sec_level >> 24;
  if ((mech & SUPPORTED_P_MECH) == 0)
    {
      mu_diag_output (MU_DIAG_NOTICE,
	      _("client requested unsupported protection mechanism (%d)"),
	      mech);
      gss_release_buffer (&min_stat, &outbuf);
      maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
      gss_release_buffer (&min_stat, &outbuf);
      return RESP_NO;
    }
  protection_mech = mech;
  client_buffer_size = sec_level & 0x00ffffffff;

  *username = malloc (outbuf.length - 4 + 1);
  if (!*username)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("not enough memory"));
      gss_release_buffer (&min_stat, &outbuf);
      maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
      gss_release_buffer (&min_stat, &outbuf);
      return RESP_NO;
    }
       
  memcpy (*username, (char *) outbuf.value + 4, outbuf.length - 4);
  (*username)[outbuf.length - 4] = '\0';
  gss_release_buffer (&min_stat, &outbuf);

  maj_stat = gss_display_name (&min_stat, client, &client_name, &mech_type);
  if (maj_stat != GSS_S_COMPLETE)
    {
      display_status ("get client name", maj_stat, min_stat);
      maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
      gss_release_buffer (&min_stat, &outbuf);
      free (*username);
      return RESP_NO;
    }

#ifdef WITH_GSS
  baduser = !gss_userok (client, *username);
#else
  baduser = imap4d_gss_userok (&client_name, *username);
#endif

  if (baduser)
    {
      mu_diag_output (MU_DIAG_NOTICE, _("GSSAPI user %s is NOT authorized as %s"),
	      (char *) client_name.value, *username);
      maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
      gss_release_buffer (&min_stat, &outbuf);
      gss_release_buffer (&min_stat, &client_name);
      free (*username);
      return RESP_NO;
    }
  else
    {
      mu_diag_output (MU_DIAG_NOTICE, _("GSSAPI user %s is authorized as %s"),
	      (char *) client_name.value, *username);
    }

  gss_release_buffer (&min_stat, &client_name);
  maj_stat = gss_delete_sec_context (&min_stat, &context, &outbuf);
  gss_release_buffer (&min_stat, &outbuf);
  return RESP_OK;
}

void
auth_gssapi_init ()
{
  auth_add ("GSSAPI", auth_gssapi);
}
    
  
