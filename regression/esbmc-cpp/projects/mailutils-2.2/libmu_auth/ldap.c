/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2008, 2009, 2010 Free Software Foundation,
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <unistd.h>
#include <stdio.h>
#include <ctype.h>

#include <mailutils/mu_auth.h>
#include <mailutils/cstr.h>

#ifdef WITH_LDAP
#include "mailutils/argcv.h"
#include "mailutils/assoc.h"
#include "mailutils/list.h"
#include "mailutils/iterator.h"
#include "mailutils/mailbox.h"
#include "mailutils/sql.h"
#include "mailutils/mu_auth.h"
#include "mailutils/error.h"
#include "mailutils/errno.h"
#include "mailutils/nls.h"
#include "mailutils/mutil.h"
#include "mailutils/stream.h"
#include "mailutils/filter.h"
#include "mailutils/md5.h"
#include "mailutils/sha1.h"
#include "mailutils/ldap.h"
#include "mailutils/vartab.h"

#include <ldap.h>
#include <lber.h>

const char *default_field_map =
"name=uid:"
"passwd=userPassword:"
"uid=uidNumber:"
"gid=gidNumber:"
"gecos=gecos:"
"dir=homeDirectory:"
"shell=loginShell";

static struct mu_ldap_module_config ldap_param;

int
mu_ldap_module_init (enum mu_gocs_op op, void *data)
{
  struct mu_ldap_module_config *cfg = data;

  if (op != mu_gocs_op_set)
    return 0;
  
  if (cfg)
    ldap_param = *cfg;

  if (ldap_param.enable)
    {
      if (!ldap_param.getpwnam_filter)
	ldap_param.getpwnam_filter = "(&(objectClass=posixAccount) (uid=%u))";
      if (!ldap_param.getpwuid_filter)
	ldap_param.getpwuid_filter =
	  "&(objectClass=posixAccount) (uidNumber=%u))";
      if (!ldap_param.field_map)
	{
	  int d;
	  mutil_parse_field_map (default_field_map, &ldap_param.field_map, &d);
	}
    }
  return 0;
}

static int
_mu_conn_setup (LDAP **pld)
{
  int rc;
  LDAPURLDesc *ludlist, **ludp;
  char **urls = NULL;
  int nurls = 0;
  char *ldapuri = NULL;
  LDAP *ld = NULL;
  int protocol = LDAP_VERSION3; /* FIXME: must be configurable */
  
  if (ldap_param.debug)
    {
      if (ber_set_option (NULL, LBER_OPT_DEBUG_LEVEL, &ldap_param.debug)
	  != LBER_OPT_SUCCESS )
	mu_error (_("cannot set LBER_OPT_DEBUG_LEVEL %d"), ldap_param.debug);

      if (ldap_set_option (NULL, LDAP_OPT_DEBUG_LEVEL, &ldap_param.debug)
	  != LDAP_OPT_SUCCESS )
	mu_error (_("could not set LDAP_OPT_DEBUG_LEVEL %d"),
		  ldap_param.debug);
    }

  if (ldap_param.url)
    {
      rc = ldap_url_parse (ldap_param.url, &ludlist);
      if (rc != LDAP_URL_SUCCESS)
	{
	  mu_error (_("cannot parse LDAP URL(s)=%s (%d)"),
		    ldap_param.url, rc);
	  return 1;
	}
      
      for (ludp = &ludlist; *ludp; )
	{
	  LDAPURLDesc *lud = *ludp;
	  char **tmp;
	  
	  if (lud->lud_dn && lud->lud_dn[0]
	      && (lud->lud_host == NULL || lud->lud_host[0] == '\0'))
	    {
	      /* if no host but a DN is provided, try DNS SRV to gather the
		 host list */
	      char	*domain = NULL, *hostlist = NULL, **hosts = NULL;
	      int hostcnt;
	      int i;
	      int len_proto = strlen(lud->lud_scheme);
	      
	      if (ldap_dn2domain (lud->lud_dn, &domain) || !domain)
		{
		  mu_error (_("DNS SRV: cannot convert DN=\"%s\" into a domain"),
			    lud->lud_dn );
		  goto dnssrv_free;
		}
	      
	      rc = ldap_domain2hostlist (domain, &hostlist);
	      if (rc)
		{
		  mu_error (_("DNS SRV: cannot convert domain=%s into a hostlist"),
			    domain);
		  goto dnssrv_free;
		}
	      
	      rc = mu_argcv_get (hostlist, " ", NULL, &hostcnt, &hosts);
	      if (rc)
		{
		  mu_error (_("DNS SRV: could not parse hostlist=\"%s\": %s"),
			    hostlist, mu_strerror (rc));
		  goto dnssrv_free;
		}
	      
	      tmp = realloc (urls, sizeof(char *) * (nurls + hostcnt + 1));
	      if (!tmp)
		{
		  mu_error ("DNS SRV %s", mu_strerror (errno));
		  goto dnssrv_free;
		}
	      
	      urls = tmp;
	      urls[nurls] = NULL;
	      
	      for (i = 0; i < hostcnt; i++)
		{
		  size_t len = len_proto + sizeof "://" - 1
		               + strlen (hosts[i])
			       + 1;

		  urls[nurls + i + 1] = NULL;
		  urls[nurls + i] = malloc (len);
		  if (!urls[nurls + i])
		    {
		      mu_error ("DNS SRV %s", mu_strerror (errno));
		      goto dnssrv_free;
		    }
		  
		  snprintf (urls[nurls + i], len, "%s://%s",
			    lud->lud_scheme, hosts[i]);
		}
	      
	      nurls += i;
	      
	    dnssrv_free:
	      mu_argcv_free (hostcnt, hosts);
	      ber_memfree (hostlist);
	      ber_memfree (domain);
	    }
	  else
	    {
	      tmp = realloc (urls, sizeof(char *) * (nurls + 2));
	      if (!tmp)
		{
		  mu_error ("DNS SRV %s", mu_strerror (errno));
		  break;
		}
	      urls = tmp;
	      urls[nurls + 1] = NULL;
	      
	      urls[nurls] = ldap_url_desc2str (lud);
	      if (!urls[nurls])
		{
		  mu_error ("DNS SRV %s", mu_strerror (errno));
		  break;
		}
	      nurls++;
	    }
	  
	  *ludp = lud->lud_next;
	  
	  lud->lud_next = NULL;
	  ldap_free_urldesc (lud);
	}

      if (ludlist)
	{
	  ldap_free_urldesc (ludlist);
	  return 1;
	}
      else if (!urls)
	return 1;
      
      rc = mu_argcv_string (nurls, urls, &ldapuri);
      if (rc)
	{
	  mu_error ("%s", mu_strerror (rc));
	  return 1;
	}
      
      ber_memvfree ((void **)urls);
    }

  mu_diag_output (MU_DIAG_INFO,
		  "constructed LDAP URI: %s", ldapuri ? ldapuri : "<DEFAULT>");

  rc = ldap_initialize (&ld, ldapuri);
  if (rc != LDAP_SUCCESS)
    {
      mu_error (_("cannot create LDAP session handle for URI=%s (%d): %s"),
		ldapuri, rc, ldap_err2string (rc));

      free (ldapuri);
      return 1;
    }
  free (ldapuri);
  
  if (ldap_param.tls)
    {
      rc = ldap_start_tls_s (ld, NULL, NULL);
      if (rc != LDAP_SUCCESS)
	{
	  mu_error (_("ldap_start_tls failed: %s"), ldap_err2string (rc));
	  return 1;
	}
    }

  ldap_set_option (ld, LDAP_OPT_PROTOCOL_VERSION, &protocol);

  /* FIXME: Timeouts, SASL, etc. */
  *pld = ld;
  return 0;
}
  
     

static int
_mu_ldap_bind (LDAP *ld)
{
  int msgid, err, rc;
  LDAPMessage *result;
  LDAPControl **ctrls;
  char msgbuf[256];
  char *matched = NULL;
  char *info = NULL;
  char **refs = NULL;
  static struct berval passwd;

  passwd.bv_val = ldap_param.passwd;
  passwd.bv_len = passwd.bv_val ? strlen (passwd.bv_val) : 0;


  msgbuf[0] = 0;

  rc = ldap_sasl_bind (ld, ldap_param.binddn, LDAP_SASL_SIMPLE, &passwd,
		       NULL, NULL, &msgid);
  if (msgid == -1)
    {
      mu_error ("ldap_sasl_bind(SIMPLE) failed: %s", ldap_err2string (rc));
      return 1;
    }

  if (ldap_result (ld, msgid, LDAP_MSG_ALL, NULL, &result ) == -1)
    {
      mu_error ("ldap_result failed");
      return 1;
    }

  rc = ldap_parse_result (ld, result, &err, &matched, &info, &refs,
			  &ctrls, 1);
  if (rc != LDAP_SUCCESS)
    {
      mu_error ("ldap_parse_result failed: %s", ldap_err2string (rc));
      return 1;
    }

  if (ctrls)
    ldap_controls_free (ctrls);

  if (err != LDAP_SUCCESS
      || msgbuf[0]
      || (matched && matched[0])
      || (info && info[0])
      || refs)
    {
      /* FIXME: Use mu_debug_t for that */
      mu_error ("ldap_bind: %s (%d)%s", ldap_err2string (err), err, msgbuf);

      if (matched && *matched) 
	mu_error ("matched DN: %s", matched);

      if (info && *info)
	mu_error ("additional info: %s", info);

      if (refs && *refs)
	{
	  int i;
	  mu_error ("referrals:");
	  for (i = 0; refs[i]; i++) 
	    mu_error ("%s", refs[i]);
	}

    }

  if (matched)
    ber_memfree (matched);
  if (info)
    ber_memfree (info);
  if (refs)
    ber_memvfree ((void **)refs);

  return err == LDAP_SUCCESS ? 0 : MU_ERR_FAILURE;
}

static void
_mu_ldap_unbind (LDAP *ld)
{
  if (ld)
    {
      ldap_set_option (ld, LDAP_OPT_SERVER_CONTROLS, NULL);
      ldap_unbind_ext (ld, NULL, NULL);
    }
}

static int 
_construct_attr_array (size_t *pargc, char ***pargv)
{
  size_t count, i;
  char **argv;
  mu_iterator_t itr = NULL;
  
  mu_assoc_count (ldap_param.field_map, &count);
  if (count == 0)
    return MU_ERR_FAILURE;
  argv = calloc (count + 1, sizeof argv[0]);

  mu_assoc_get_iterator (ldap_param.field_map, &itr);
  for (i = 0, mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr), i++)
    {
      char **str;
      mu_iterator_current (itr, (void**) &str); 
      argv[i] = strdup (*str);
    }
  mu_iterator_destroy (&itr);
  argv[i] = NULL;
  
  *pargc = count;
  *pargv = argv;
  
  return 0;
}

/* FIXME: Duplicated in other modules */
static void
get_quota (mu_off_t *pquota, const char *str)
{
  char *p;
  mu_off_t quota = strtoul (str, &p, 10);
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
      mu_error (_("invalid value for quota: %s"), str);
    }
}

static void
_free_partial_auth_data (struct mu_auth_data *d)
{
  free (d->name);
  free (d->passwd);
  free (d->gecos);
  free (d->dir);
  free (d->shell);
  free (d->mailbox);
}

static int
_mu_entry_to_auth_data (LDAP *ld, LDAPMessage *msg,
			struct mu_auth_data **return_data)
{
  int rc;
  BerElement *ber = NULL;
  struct berval bv;
  char *ufn = NULL;
  struct mu_auth_data d;
  mu_iterator_t itr = NULL;
  
  memset (&d, 0, sizeof d);
  
  /* FIXME: should use mu_debug_t */
  rc = ldap_get_dn_ber (ld, msg, &ber, &bv);
  ufn = ldap_dn2ufn (bv.bv_val);
  mu_error ("INFO: %s", ufn);
  ldap_memfree (ufn);
  
  mu_assoc_get_iterator (ldap_param.field_map, &itr);
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      char *key;
      char **pattr;
      char *attr;
      struct berval **values;
      
      mu_iterator_current_kv (itr, (const void **)&key, (void**) &pattr);
      attr = *pattr;
      values = ldap_get_values_len (ld, msg, attr);
      if (!values || !values[0])
	{
	  mu_error ("LDAP field `%s' (`%s') has NULL value",
		    key, *pattr);
	  _free_partial_auth_data (&d);
	  return MU_ERR_READ;
	}
      
      if (strcmp (key, MU_AUTH_NAME) == 0)
	d.name = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_PASSWD) == 0)
	d.passwd = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_UID) == 0)
	d.uid = atoi (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_GID) == 0)
	d.gid = atoi (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_GECOS) == 0)
	d.gecos = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_DIR) == 0)
	d.dir = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_SHELL) == 0)   
	d.shell = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_MAILBOX) == 0)
	d.mailbox = strdup (values[0]->bv_val);
      else if (strcmp (key, MU_AUTH_QUOTA) == 0)   
	get_quota (&d.quota, values[0]->bv_val);
      
      ldap_value_free_len (values);
    }
  
  rc = mu_auth_data_alloc (return_data,
			   d.name,
			   d.passwd,
			   d.uid,
			   d.gid,
			   d.gecos,
			   d.dir,
			   d.shell,
			   d.mailbox,
			   1);
  if (rc == 0)
    mu_auth_data_set_quota (*return_data, d.quota);

  _free_partial_auth_data (&d);
  
  return rc;
}
    

static int
_mu_ldap_search (LDAP *ld, const char *filter_pat, const char *key,
		 struct mu_auth_data **return_data)
{
  int rc;
  char *filter;
  char **attrs;
  size_t nattrs;
  LDAPMessage *res, *msg;
  ber_int_t msgid;
  mu_vartab_t vtab;

  rc = _construct_attr_array (&nattrs, &attrs);
  if (rc)
    return rc;

  mu_vartab_create (&vtab);
  mu_vartab_define (vtab, "user", key, 1);
  mu_vartab_define (vtab, "u", key, 1);
  rc = mu_vartab_expand (vtab, filter_pat, &filter);
  mu_vartab_destroy (&vtab);
  if (rc)
    {
      mu_argcv_free (nattrs, attrs);
      return ENOMEM;
    }

  rc = ldap_search_ext (ld, ldap_param.base, LDAP_SCOPE_SUBTREE,
			filter, attrs, 0,
			NULL, NULL, NULL, -1, &msgid);
  mu_argcv_free (nattrs, attrs);
  free (filter);

  if (rc != LDAP_SUCCESS)
    {
      mu_error ("ldap_search_ext: %s", ldap_err2string (rc));
      if (rc == LDAP_NO_SUCH_OBJECT)
	return MU_ERR_NOENT;
      else
	return MU_ERR_FAILURE;
    }

  rc = ldap_result (ld, msgid, LDAP_MSG_ALL, NULL, &res );
  if (rc < 0)
    {
      mu_error ("ldap_result failed");
      return MU_ERR_FAILURE;
    }

  rc = ldap_count_entries (ld, res);
  if (rc == 0)
    {
      mu_error ("not enough entires");
      return MU_ERR_NOENT;
    }
  if (rc > 1)
    mu_error ("LDAP: too many entries for key %s", key);
      

  msg = ldap_first_entry (ld, res);
  rc = _mu_entry_to_auth_data (ld, msg, return_data);
  ldap_msgfree (res);
  
  return rc;
}



typedef int (*pwcheck_fp) (const char *, const char *);


static int
chk_crypt (const char *db_pass, const char *pass)
{
  return strcmp (db_pass, crypt (pass, db_pass)) == 0 ?
              0 : MU_ERR_AUTH_FAILURE;
}

static int
chk_md5 (const char *db_pass, const char *pass)
{
  unsigned char md5digest[16];
  unsigned char d1[16];
  struct mu_md5_ctx md5context;
  mu_stream_t str = NULL, flt = NULL;
  
  mu_md5_init_ctx (&md5context);
  mu_md5_process_bytes (pass, strlen (pass), &md5context);
  mu_md5_finish_ctx (&md5context, md5digest);

  mu_memory_stream_create (&str, NULL, MU_STREAM_NO_CHECK);
  mu_filter_create (&flt, str, "base64", MU_FILTER_DECODE,
		    MU_STREAM_READ | MU_STREAM_NO_CHECK);
  mu_stream_open (str);
  mu_stream_sequential_write (str, db_pass, strlen (db_pass));

  mu_stream_read (flt, (char*) d1, sizeof d1, 0, NULL);
  mu_stream_destroy (&flt, NULL);
  
  return memcmp (md5digest, d1, sizeof md5digest) == 0 ?
                  0 : MU_ERR_AUTH_FAILURE;
}

static int
chk_smd5 (const char *db_pass, const char *pass)
{
  int rc;
  unsigned char md5digest[16];
  unsigned char *d1;
  struct mu_md5_ctx md5context;
  mu_stream_t str = NULL, flt = NULL;
  size_t size;
  
  mu_memory_stream_create (&str, NULL, MU_STREAM_NO_CHECK);
  mu_filter_create (&flt, str, "base64", MU_FILTER_DECODE,
		    MU_STREAM_READ | MU_STREAM_NO_CHECK);
  mu_stream_open (str);
  size = strlen (db_pass);
  mu_stream_sequential_write (str, db_pass, size);

  d1 = malloc (size);
  if (!d1)
    {
      mu_stream_destroy (&flt, NULL);
      return ENOMEM;
    }
  
  mu_stream_read (flt, (char*) d1, size, 0, &size);
  mu_stream_destroy (&flt, NULL);

  if (size <= 16)
    {
      mu_error ("malformed SMD5 password: %s", db_pass);
      return MU_ERR_FAILURE;
    }
  
  mu_md5_init_ctx (&md5context);
  mu_md5_process_bytes (pass, strlen (pass), &md5context);
  mu_md5_process_bytes (d1 + 16, size - 16, &md5context);
  mu_md5_finish_ctx (&md5context, md5digest);

  rc = memcmp (md5digest, d1, sizeof md5digest) == 0 ?
                  0 : MU_ERR_AUTH_FAILURE;
  free (d1);
  return rc;
}

static int
chk_sha (const char *db_pass, const char *pass)
{
  unsigned char sha1digest[20];
  unsigned char d1[20];
  mu_stream_t str = NULL, flt = NULL;
  struct mu_sha1_ctx sha1context;
   
  mu_sha1_init_ctx (&sha1context);
  mu_sha1_process_bytes (pass, strlen (pass), &sha1context);
  mu_sha1_finish_ctx (&sha1context, sha1digest);

  mu_memory_stream_create (&str, NULL, MU_STREAM_NO_CHECK);
  mu_filter_create (&flt, str, "base64", MU_FILTER_DECODE,
		    MU_STREAM_READ | MU_STREAM_NO_CHECK);
  mu_stream_open (str);
  mu_stream_sequential_write (str, db_pass, strlen (db_pass));

  mu_stream_read (flt, (char*) d1, sizeof d1, 0, NULL);
  mu_stream_destroy (&flt, NULL);
  
  return memcmp (sha1digest, d1, sizeof sha1digest) == 0 ?
                  0 : MU_ERR_AUTH_FAILURE;
}

static int
chk_ssha (const char *db_pass, const char *pass)
{
  int rc;
  unsigned char sha1digest[20];
  unsigned char *d1;
  struct mu_sha1_ctx sha1context;
  mu_stream_t str = NULL, flt = NULL;
  size_t size;
  
  mu_memory_stream_create (&str, NULL, MU_STREAM_NO_CHECK);
  mu_filter_create (&flt, str, "base64", MU_FILTER_DECODE,
		    MU_STREAM_READ | MU_STREAM_NO_CHECK);
  mu_stream_open (str);
  size = strlen (db_pass);
  mu_stream_sequential_write (str, db_pass, size);

  d1 = malloc (size);
  if (!d1)
    {
      mu_stream_destroy (&flt, NULL);
      return ENOMEM;
    }
  
  mu_stream_read (flt, (char*) d1, size, 0, &size);
  mu_stream_destroy (&flt, NULL);
  mu_stream_destroy (&str, NULL);

  if (size <= 16)
    {
      mu_error ("malformed SSHA1 password: %s", db_pass);
      return MU_ERR_FAILURE;
    }
  
  mu_sha1_init_ctx (&sha1context);
  mu_sha1_process_bytes (pass, strlen (pass), &sha1context);
  mu_sha1_process_bytes (d1 + 20, size - 20, &sha1context);
  mu_sha1_finish_ctx (&sha1context, sha1digest);

  rc = memcmp (sha1digest, d1, sizeof sha1digest) == 0 ?
                  0 : MU_ERR_AUTH_FAILURE;
  free (d1);
  return rc;
}

static struct passwd_algo
{
  char *algo;
  size_t len;
  pwcheck_fp pwcheck;
} pwtab[] = {
#define DP(s, f) { #s, sizeof (#s) - 1, f }
  DP (CRYPT, chk_crypt),
  DP (MD5, chk_md5),
  DP (SMD5, chk_smd5),
  DP (SHA, chk_sha),
  DP (SSHA, chk_ssha),
  { NULL }
#undef DP
};

static pwcheck_fp
find_pwcheck (const char *algo, int len)
{
  struct passwd_algo *p;
  for (p = pwtab; p->algo; p++)
    if (len == p->len && mu_c_strncasecmp (p->algo, algo, len) == 0)
      return p->pwcheck;
  return NULL;
}

static int
mu_ldap_authenticate (struct mu_auth_data **return_data MU_ARG_UNUSED,
		      const void *key,
		      void *func_data MU_ARG_UNUSED, void *call_data)
{
  const struct mu_auth_data *auth_data = key;
  char *db_pass = auth_data->passwd;
  char *pass = call_data;

  if (auth_data->passwd == NULL || !pass)
    return EINVAL;

  if (db_pass[0] == '{')
    {
      int len;
      char *algo;
      pwcheck_fp pwcheck;

      
      algo = db_pass + 1;
      for (len = 0; algo[len] != '}'; len++)
	if (algo[len] == 0)
	  {
	    /* Possibly malformed password, try plaintext anyway */
	    return strcmp (db_pass, pass) == 0 ? 0 : MU_ERR_FAILURE;
	  }
      db_pass = algo + len + 1;
      pwcheck = find_pwcheck (algo, len);
      if (pwcheck)
	return pwcheck (db_pass, pass);
      else
	{
	  mu_error ("Unsupported password algorithm scheme: %.*s",
		    len, algo);
	  return MU_ERR_FAILURE;
	}
    }
  
  return strcmp (db_pass, pass) == 0 ? 0 : MU_ERR_AUTH_FAILURE;
}


static int
mu_auth_ldap_user_by_name (struct mu_auth_data **return_data,
			   const void *key,
			   void *func_data MU_ARG_UNUSED,
			   void *call_data MU_ARG_UNUSED)
{
  int rc;
  LDAP *ld;

  if (!ldap_param.enable)
    return ENOSYS;
  if (_mu_conn_setup (&ld))
    return MU_ERR_FAILURE;
  if (_mu_ldap_bind (ld))
    return MU_ERR_FAILURE;
  rc = _mu_ldap_search (ld, ldap_param.getpwnam_filter, key, return_data);
  _mu_ldap_unbind (ld);
  return rc;
}

static int
mu_auth_ldap_user_by_uid (struct mu_auth_data **return_data,
			  const void *key,
			  void *func_data MU_ARG_UNUSED,
			  void *call_data MU_ARG_UNUSED)
{
  int rc;
  LDAP *ld;
  char uidstr[128];

  if (!ldap_param.enable)
    return ENOSYS;
  if (_mu_conn_setup (&ld))
    return MU_ERR_FAILURE;
  if (_mu_ldap_bind (ld))
    return MU_ERR_FAILURE;

  snprintf (uidstr, sizeof (uidstr), "%u", *(uid_t*)key);
  rc = _mu_ldap_search (ld, ldap_param.getpwuid_filter, uidstr, return_data);
  _mu_ldap_unbind (ld);
  return rc;
}


#else
# define mu_ldap_module_init NULL
# define mu_ldap_authenticate mu_auth_nosupport
# define mu_auth_ldap_user_by_name mu_auth_nosupport 
# define mu_auth_ldap_user_by_uid mu_auth_nosupport
#endif

struct mu_auth_module mu_auth_ldap_module = {
  "ldap",
  mu_ldap_module_init,
  mu_ldap_authenticate,
  NULL,
  mu_auth_ldap_user_by_name,
  NULL,
  mu_auth_ldap_user_by_uid,
  NULL
};
