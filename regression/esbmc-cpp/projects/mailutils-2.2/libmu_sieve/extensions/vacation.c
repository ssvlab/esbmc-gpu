/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2009, 2010 Free Software Foundation, Inc.

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

/* Syntax: vacation [:days <ndays: number>]
                    [:subject <subject: string>]
		    [:aliases <address-list: list>]
		    [:addresses <noreply-address-list: list>]
		    [:reply_regex <expr: string>]
		    [:reply_prefix <prefix: string>]
		    <reply text: string>
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <signal.h>
#include <regex.h>
#include <mu_dbm.h>
#include <mailutils/sieve.h>
#include <mailutils/mu_auth.h>

/* Build a mime response message from original message MSG. TEXT
   is the message text.
   FIXME: This is for future use, when I add :mime tag
*/
static int
build_mime (mu_sieve_machine_t mach, mu_list_t tags, mu_mime_t *pmime,
	    mu_message_t msg, const char *text)
{
  mu_mime_t mime = NULL;
  mu_message_t newmsg;
  mu_stream_t stream, input, save_input = NULL;
  mu_header_t hdr;
  mu_body_t body;
  char buf[512];
  size_t n;
  char *header = "Content-Type: text/plain;charset=" MU_SIEVE_CHARSET "\n"
 	       "Content-Transfer-Encoding: 8bit\n\n";
  int rc;
  
  mu_mime_create (&mime, NULL, 0);
  mu_message_create (&newmsg, NULL);
  mu_message_get_body (newmsg, &body);
  mu_body_get_stream (body, &stream);

  if ((rc = mu_memory_stream_create (&input, 0, MU_STREAM_RDWR)))
    {
      mu_sieve_error (mach,
		      _("cannot create temporary stream: %s"),
		      mu_strerror (rc));
      return 1;
    }

  if ((rc = mu_stream_open (input)))
    {
      mu_sieve_error (mach,
		      _("cannot open temporary stream: %s"),
		      mu_strerror (rc));
      return 1;
    }
  
  mu_stream_write (input, text, strlen (text), 0, &n);

  if (mu_sieve_tag_lookup (tags, "mime", NULL))
    {
      mu_stream_t fstr;
      rc = mu_filter_create (&fstr, input, "base64",
			     MU_FILTER_ENCODE, MU_STREAM_READ);
      if (rc == 0) 
	{
	  header = "Content-Type: text/plain;charset=" MU_SIEVE_CHARSET "\n"
	           "Content-Transfer-Encoding: base64\n\n";
	  save_input = input;
	  input = fstr;
	}
    }

  while (rc == 0
	 && mu_stream_sequential_read (input, buf, sizeof buf, &n) == 0
	 && n > 0)
    rc = mu_stream_sequential_write (stream, buf, n);

  mu_stream_destroy (&input, mu_stream_get_owner (input));
  if (save_input)
    mu_stream_destroy (&save_input, mu_stream_get_owner (save_input));
  
  mu_header_create (&hdr, header, strlen (header), newmsg);
  mu_message_set_header (newmsg, hdr, NULL);

  mu_mime_add_part (mime, newmsg);
  mu_message_unref (newmsg);

  *pmime = mime;

  return 0;
}



/* Produce diagnostic output. */
static int
diag (mu_sieve_machine_t mach)
{
  if (mu_sieve_get_debug_level (mach) & MU_SIEVE_DEBUG_TRACE)
    {
      mu_sieve_locus_t locus;
      mu_sieve_get_locus (mach, &locus);
      mu_sieve_debug (mach, "%s:%lu: VACATION\n",
		   locus.source_file,
		   (unsigned long) locus.source_line);
    }

  mu_sieve_log_action (mach, "VACATION", NULL);
  return mu_sieve_is_dry_run (mach);
}


struct addr_data {
  mu_address_t addr;
  char *my_address;
};

static int
_compare (void *item, void *data)
{
  struct addr_data *ad = data;
  int rc = mu_address_contains_email (ad->addr, item);
  if (rc)
    ad->my_address = item;
  return rc;
}

/* Check whether an alias from ADDRESSES is part of To: or Cc: headers
   of the originating mail. Return non-zero if so and store a pointer
   to the matching address to *MY_ADDRESS. */
static int
match_addresses (mu_header_t hdr, mu_sieve_value_t *addresses, char **my_address)
{
  int match = 0;
  char *str;
  struct addr_data ad;

  ad.my_address = NULL;
  if (mu_header_aget_value (hdr, MU_HEADER_TO, &str) == 0)
    {
      if (!mu_address_create (&ad.addr, str))
	{
	  match += mu_sieve_vlist_do (addresses, _compare, &ad);
	  mu_address_destroy (&ad.addr);
	}
      free (str);
    }

  if (!match && mu_header_aget_value (hdr, MU_HEADER_CC, &str) == 0)
    {
      if (!mu_address_create (&ad.addr, str))
	{
	  match += mu_sieve_vlist_do (addresses, _compare, &ad);
	  mu_address_destroy (&ad.addr);
	}
      free (str);
    }
  *my_address = ad.my_address;
  return match;
}


struct regex_data {
  mu_sieve_machine_t mach;
  char *email;
};

static int
regex_comparator (void *item, void *data)
{
  regex_t preg;
  int rc;
  struct regex_data *d = data;
  
  if (regcomp (&preg, item,
	       REG_EXTENDED | REG_NOSUB | REG_NEWLINE | REG_ICASE))
    {
      mu_sieve_error (d->mach,
		      _("%lu: cannot compile regular expression \"%s\""),
		      (unsigned long) mu_sieve_get_message_num (d->mach),
		      (char*) item);
      return 0;
    }
  rc = regexec (&preg, d->email, 0, NULL, 0) == 0;
  regfree (&preg);
  return rc;
}

/* Decide whether EMAIL address should not be responded to.
 */
static int
noreply_address_p (mu_sieve_machine_t mach, mu_list_t tags, char *email)
{
  int i, rc = 0;
  mu_sieve_value_t *arg;
  struct regex_data rd;
  static char *noreply_sender[] = {
    ".*-REQUEST@.*",
    ".*-RELAY@.*",
    ".*-OWNER@.*",
    "^OWNER-.*",
    "^postmaster@.*",
    "^UUCP@.*",
    "^MAILER@.*",
    "^MAILER-DAEMON@.*",
    NULL
  };

  rd.mach = mach;
  rd.email = email;
  for (i = 0; rc == 0 && noreply_sender[i]; i++)
    rc = regex_comparator (noreply_sender[i], &rd);

  if (!rc && mu_sieve_tag_lookup (tags, "addresses", &arg))
    rc = mu_sieve_vlist_do (arg, regex_comparator, &rd);
  
  return rc;
}


/* Return T if letter precedence is 'bulk' or 'junk' */
static int
bulk_precedence_p (mu_header_t hdr)
{
  int rc = 0;
  char *str;
  if (mu_header_aget_value (hdr, MU_HEADER_PRECEDENCE, &str) == 0)
    {
      rc = mu_c_strcasecmp (str, "bulk") == 0
	   || mu_c_strcasecmp (str, "junk") == 0;
      free (str);
    }
  return rc;
}

#define	DAYS_MIN	1
#define	DAYS_DEFAULT	7
#define	DAYS_MAX	60

/* Check and updated vacation database. Return 0 if the mail should
   be answered. */
static int
check_db (mu_sieve_machine_t mach, mu_list_t tags, char *from)
{
#ifdef USE_DBM
  DBM_FILE db;
  DBM_DATUM key, value;
  time_t now;
  char buffer[64];
  char *file, *home;
  mu_sieve_value_t *arg;
  unsigned int days;
  int rc;
  
  if (mu_sieve_tag_lookup (tags, "days", &arg))
    {
      days = arg->v.number;
      if (days < DAYS_MIN)
	days = DAYS_MIN;
      else if (days > DAYS_MAX)
	days = DAYS_MAX;
    }
  else
    days = DAYS_DEFAULT;

  home = mu_get_homedir ();

  if (asprintf (&file, "%s/.vacation", (home ? home : ".")) == -1)
    {
      mu_sieve_error (mach, _("%lu: cannot build db file name"),
		      (unsigned long) mu_sieve_get_message_num (mach));
      free (home);
      mu_sieve_abort (mach);
    }
  free (home);
  
  rc = mu_dbm_open (file, &db, MU_STREAM_RDWR, 0600);
  if (rc)
    {
      mu_sieve_error (mach,
		      _("%lu: cannot open `%s': %s"),
		      (unsigned long) mu_sieve_get_message_num (mach), file,
		      mu_strerror (rc));
      free (file);
      mu_sieve_abort (mach);
    }
  free (file);

  time (&now);

  MU_DATUM_SIZE (key) = strlen (from);
  MU_DATUM_PTR (key) = from;

  rc = mu_dbm_fetch (db, key, &value);
  if (!rc)
    {
      time_t last;
      
      strncpy(buffer, MU_DATUM_PTR (value), MU_DATUM_SIZE (value));
      buffer[MU_DATUM_SIZE (value)] = 0;

      last = (time_t) strtoul (buffer, NULL, 0);


      if (last + (24 * 60 * 60 * days) > now)
	{
	  mu_dbm_close (db);
	  return 1;
	}
    }

  MU_DATUM_SIZE (value) = snprintf (buffer, sizeof buffer, "%ld", now);
  MU_DATUM_PTR (value) = buffer;
  MU_DATUM_SIZE (key) = strlen (from);
  MU_DATUM_PTR (key) = from;

  mu_dbm_insert (db, key, value, 1);
  mu_dbm_close (db);
  return 0;
#else
  mu_sieve_error (mach,
	       /* TRANSLATORS: 'vacation' and ':days' are Sieve keywords.
		  Do not translate them! */
	       _("%d: vacation compiled without DBM support. Ignoring :days tag"),
	       mu_sieve_get_message_num (mach));
  return 0;
#endif
}

/* Add a reply prefix to the subject. *PSUBJECT points to the
   original subject, which must be allocated using malloc. Before
   returning its value is freed and replaced with the new one.
   Default reply prefix is "Re: ", unless overridden by
   "reply_prefix" tag.
 */
static void
re_subject (mu_sieve_machine_t mach, mu_list_t tags, char **psubject)
{
  char *subject;
  mu_sieve_value_t *arg;
  char *prefix = "Re";
  
  if (mu_sieve_tag_lookup (tags, "reply_prefix", &arg))
    {
      prefix = arg->v.string;
    }

  subject = malloc (strlen (*psubject) + strlen (prefix) + 3);
  if (!subject)
    {
      mu_sieve_error (mach,
		      _("%lu: not enough memory"),
		      (unsigned long) mu_sieve_get_message_num (mach));
      return;
    }
  
  strcpy (subject, prefix);
  strcat (subject, ": ");
  strcat (subject, *psubject);
  free (*psubject);
  *psubject = subject;
}

/* Process reply subject header.

   If :subject is set, its value is used.
   Otherwise, if original subject matches reply_regex, it is used verbatim.
   Otherwise, reply_prefix is prepended to it. */

static void
vacation_subject (mu_sieve_machine_t mach, mu_list_t tags,
		  mu_message_t msg, mu_header_t newhdr)
{
  mu_sieve_value_t *arg;
  char *value;
  char *subject;
  int subject_allocated = 0;
  mu_header_t hdr;
  
  if (mu_sieve_tag_lookup (tags, "subject", &arg))
    subject =  arg->v.string;
  else if (mu_message_get_header (msg, &hdr) == 0
	   && mu_header_aget_value_unfold (hdr, MU_HEADER_SUBJECT, &value) == 0)
    {
      char *p;
      
      int rc = mu_rfc2047_decode (MU_SIEVE_CHARSET, value, &p);

      subject_allocated = 1;
      if (rc)
	{
	  subject = value;
	  value = NULL;
	}
      else
	{
	  subject = p;
	}

      if (mu_sieve_tag_lookup (tags, "reply_regex", &arg))
	{
	  char *err = NULL;
	  
	  rc = mu_unre_set_regex (arg->v.string, 0, &err);
	  if (rc)
	    {
	      mu_sieve_error (mach,
			      /* TRANSLATORS: 'vacation' is the name of the
				 Sieve action. Do not translate it! */
			      _("%lu: vacation - cannot compile reply prefix regexp: %s: %s"),
			      (unsigned long) mu_sieve_get_message_num (mach),
			      mu_strerror (rc),
			      err ? err : "");
	    }
	}
	  
      if (mu_unre_subject (subject, NULL))
	re_subject (mach, tags, &subject);
      
      free (value);
    }
  else
    subject = "Re: Your mail";
    
  if (mu_rfc2047_encode (MU_SIEVE_CHARSET, "quoted-printable",
			 subject, &value))
    mu_header_set_value (newhdr, MU_HEADER_SUBJECT, subject, 0);
  else
    {
      mu_header_set_value (newhdr, MU_HEADER_SUBJECT, value, 0);
      free (value);
    }

  if (subject_allocated)
    free (subject);
}

/* Generate and send the reply message */
static int
vacation_reply (mu_sieve_machine_t mach, mu_list_t tags, mu_message_t msg,
		char *text, char *to, char *from)
{
  mu_mime_t mime = NULL;
  mu_message_t newmsg;
  mu_header_t newhdr;
  mu_address_t to_addr = NULL, from_addr = NULL;
  char *value;
  mu_mailer_t mailer;
  int rc;
  
  if (build_mime (mach, tags, &mime, msg, text))
    return -1;
  mu_mime_get_message (mime, &newmsg);
  mu_message_get_header (newmsg, &newhdr);
  
  rc = mu_address_create (&to_addr, to);
  if (rc)
    {
      mu_sieve_error (mach,
		      _("%lu: cannot create recipient address <%s>: %s"),
		      (unsigned long) mu_sieve_get_message_num (mach),
		      from, mu_strerror (rc));
    }
  else
    {
      mu_header_set_value (newhdr, MU_HEADER_TO, to, 0);
      
      vacation_subject (mach, tags, msg, newhdr);
      
      if (from)
        {
          if (mu_address_create (&from_addr, from))
	    from_addr = NULL;
        }
      else
        {
          from_addr = NULL;
        }
      
      if (mu_rfc2822_in_reply_to (msg, &value) == 0)
        {
          mu_header_set_value (newhdr, MU_HEADER_IN_REPLY_TO, value, 1);
          free (value);
        }
      
      if (mu_rfc2822_references (msg, &value) == 0)
        {
          mu_header_set_value (newhdr, MU_HEADER_REFERENCES, value, 1);
          free (value);
        }
      
      mailer = mu_sieve_get_mailer (mach);
      rc = mu_mailer_open (mailer, 0);
      if (rc)
	{
	  mu_url_t url = NULL;
	  mu_mailer_get_url (mailer, &url);
      
	  mu_sieve_error (mach,
			  _("%lu: cannot open mailer %s: %s"),
			  (unsigned long) mu_sieve_get_message_num (mach),
			  mu_url_to_string (url), mu_strerror (rc));
	}
      else
	{
	  rc = mu_mailer_send_message (mailer, newmsg, from_addr, to_addr);
	  mu_mailer_close (mailer);
	}
      mu_mailer_destroy (&mailer);
    }
  mu_address_destroy (&to_addr);
  mu_address_destroy (&from_addr);
  mu_mime_destroy (&mime);
  return rc;
}

int
sieve_action_vacation (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  int rc;
  char *text, *from;
  mu_sieve_value_t *val;
  mu_message_t msg;
  mu_header_t hdr;
  char *my_address = mu_sieve_get_daemon_email (mach);
  
  if (diag (mach))
    return 0;
  
  val = mu_sieve_value_get (args, 0);
  if (!val)
    {
      mu_sieve_error (mach, _("cannot get text!"));
      mu_sieve_abort (mach);
    }
  else
    text = val->v.string;

  msg = mu_sieve_get_message (mach);
  mu_message_get_header (msg, &hdr);

  if (mu_sieve_tag_lookup (tags, "sender", &val))
    {
      /* Debugging hook: :sender sets fake reply address */
      from = strdup (val->v.string);
    }
  else if (mu_sieve_get_message_sender (msg, &from))
    {
      mu_sieve_error (mach,
		      _("%lu: cannot get sender address"),
		      (unsigned long) mu_sieve_get_message_num (mach));
      mu_sieve_abort (mach);
    }

  if (mu_sieve_tag_lookup (tags, "aliases", &val)
      && match_addresses (hdr, val, &my_address) == 0)
    return 0;

  if (noreply_address_p (mach, tags, from)
      || bulk_precedence_p (hdr)
      || check_db (mach, tags, from))
    {
      free (from);
      return 0;
    }

  rc = vacation_reply (mach, tags, msg, text, from, my_address);
  free (from);
  if (rc == -1)
    mu_sieve_abort (mach);
  return rc;
}

/* Tagged arguments: */
static mu_sieve_tag_def_t vacation_tags[] = {
  {"days", SVT_NUMBER},
  {"subject", SVT_STRING},
  {"aliases", SVT_STRING_LIST},
  {"addresses", SVT_STRING_LIST},
  {"reply_regex", SVT_STRING},
  {"reply_prefix", SVT_STRING},
  {"mime", SVT_VOID},
  {NULL}
};

static mu_sieve_tag_group_t vacation_tag_groups[] = {
  {vacation_tags, NULL},
  {NULL}
};

/* Required arguments: */
static mu_sieve_data_type vacation_args[] = {
  SVT_STRING,			/* message text */
  SVT_VOID
};

int SIEVE_EXPORT (vacation, init) (mu_sieve_machine_t mach)
{
  return mu_sieve_register_action (mach, "vacation", sieve_action_vacation,
				vacation_args, vacation_tag_groups, 1);
}
