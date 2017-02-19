/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library.  If not,
   see <http://www.gnu.org/licenses/>. */

/* This file provides backward-compatible "remote+" mailbox types,
   introduced in v. 2.0.

   They are only used by maidag.

   This file will be removed in v. 2.2
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/diag.h>

#include <url0.h>
#include <mailer0.h>
#include <registrar0.h>

#ifdef ENABLE_SMTP

static int
_url_remote_init (mu_url_t url, const char *new_scheme)
{
  char *scheme;
  mu_record_t record;
  int rc;
  
  mu_diag_output (MU_DIAG_WARNING,
		  "%s: this URL scheme is deprecated, use %s instead",
		  url->name, new_scheme);

  rc = mu_registrar_lookup_scheme (new_scheme, &record);
  if (rc)
    return rc;

  scheme = strdup (new_scheme);
  if (!scheme)
    return ENOMEM;
  
  free (url->scheme);
  url->scheme = scheme;
  
  return record->_url ? record->_url (url) : 0;
}


static int
_url_remote_smtp_init (mu_url_t url)
{
  return _url_remote_init (url, "smtp");
}

static struct _mu_record _mu_remote_smtp_record = {
  MU_SMTP_PRIO,
  "remote+smtp",
  _url_remote_smtp_init,	/* url init.  */
  _mu_mailer_mailbox_init,      /* Mailbox init.  */
  NULL,		                /* Mailer init.  */
  _mu_mailer_folder_init,	/* Folder init.  */
  NULL,				/* No need for a back pointer.  */
  NULL,				/* _is_scheme method.  */
  NULL,				/* _get_url method.  */
  NULL,				/* _get_mailbox method.  */
  NULL,				/* _get_mailer method.  */
  NULL				/* _get_folder method.  */
};

mu_record_t mu_remote_smtp_record = &_mu_remote_smtp_record;
#else
mu_record_t mu_remote_smtp_record = NULL;
#endif


#ifdef ENABLE_SENDMAIL
static int
_url_remote_sendmail_init (mu_url_t url)
{
  return _url_remote_init (url, "sendmail");
}

static struct _mu_record _mu_remote_sendmail_record =
{
  MU_SENDMAIL_PRIO,
  "remote+sendmail",
  _url_remote_sendmail_init,    /* url init.  */
  _mu_mailer_mailbox_init,      /* Mailbox entry.  */
  _mu_mailer_sendmail_init, /* Mailer entry.  */
  _mu_mailer_folder_init, /* Folder entry.  */
  NULL, /* No need for a back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};


mu_record_t mu_remote_sendmail_record = &_mu_remote_sendmail_record;


static int
_url_remote_prog_init (mu_url_t url)
{
  return _url_remote_init (url, "prog");
}

static struct _mu_record _mu_remote_prog_record =
{
  MU_PROG_PRIO,
  "remote+prog",
  _url_remote_prog_init,    /* url init.  */
  _mu_mailer_mailbox_init,  /* Mailbox entry.  */
  _mu_mailer_prog_init, /* Mailer entry.  */
  _mu_mailer_folder_init, /* Folder entry.  */
  NULL, /* No need for a back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};

mu_record_t mu_remote_prog_record = &_mu_remote_prog_record;

#else
mu_record_t mu_remote_sendmail_record = NULL;
mu_record_t mu_remote_prog_record = NULL;
#endif
