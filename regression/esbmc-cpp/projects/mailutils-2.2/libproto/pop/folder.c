/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#ifdef ENABLE_POP

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/auth.h>
#include <mailutils/errno.h>
#include <mailutils/mailbox.h>
#include <mailutils/cstr.h>
#include <mailutils/cctype.h>

#include <folder0.h>
#include <registrar0.h>
#include <url0.h>

/* We export url parsing and the initialisation of
   the mailbox, via the register entry/record.  */

static struct _mu_record _pop_record =
{
  MU_POP_PRIO,
  MU_POP_SCHEME,
  _url_pop_init, /* Url init.  */
  _mailbox_pop_init, /* Mailbox init.  */
  NULL, /* Mailer init.  */
  _folder_pop_init, /* Folder init.  */
  NULL, /* No need for an back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};
mu_record_t mu_pop_record = &_pop_record;

#ifdef WITH_TLS
static struct _mu_record _pops_record =
{
  MU_POP_PRIO,
  MU_POPS_SCHEME,
  _url_pops_init, /* Url init.  */
  _mailbox_pops_init, /* Mailbox init.  */
  NULL, /* Mailer init.  */
  _folder_pop_init, /* Folder init.  */
  NULL, /* No need for an back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};
mu_record_t mu_pops_record = &_pops_record;
#else
mu_record_t mu_pops_record = NULL;
#endif /* WITH_TLS */

static int folder_pop_open  (mu_folder_t, int);
static int folder_pop_close (mu_folder_t);
static int folder_pop_get_authority (mu_folder_t, mu_authority_t *);
extern int _pop_user         (mu_authority_t);
extern int _pop_apop         (mu_authority_t);

/* XXX: The way, the POP folder is handled is not clean at all.
   the I/O functions should have been here on folder, not in  mbx_pop.c  */
int
_folder_pop_init (mu_folder_t folder)
{
  int status;

  /* Set the authority early:
     (1) so we can check for errors.
     (2) allow the client to get the authority for setting the ticket
     before the open.  */
  status = folder_pop_get_authority (folder, NULL);
  if (status != 0)
    return status;

  folder->_open = folder_pop_open;
  folder->_close = folder_pop_close;
  return 0;
}

static int
folder_pop_open (mu_folder_t folder, int flags)
{
  mu_mailbox_t mbox = folder->data;
  return mu_mailbox_open (mbox, flags);
}

static int
folder_pop_close (mu_folder_t folder)
{
  mu_mailbox_t mbox = folder->data;
  return mu_mailbox_close (mbox);
}

static int
folder_pop_get_authority (mu_folder_t folder, mu_authority_t *pauth)
{
  int status = 0;
  if (folder->authority == NULL)
    {
      /* assert (folder->url); */
      if (folder->url == NULL)
	return EINVAL;

      if (folder->url->auth == NULL
	  || strcmp (folder->url->auth, "*") == 0)
	{
	  status = mu_authority_create (&folder->authority, NULL, folder);
	  mu_authority_set_authenticate (folder->authority, _pop_user, folder);
	}
      /*
	"+apop" could be supported.
	Anything else starting with "+" is an extension mechanism.
	Without a "+" it's a SASL mechanism.
      */
      else if (mu_c_strcasecmp (folder->url->auth, "+APOP") == 0)
	{
	  status = mu_authority_create (&folder->authority, NULL, folder);
	  mu_authority_set_authenticate (folder->authority, _pop_apop, folder);
	}
      else
	{
	  status = MU_ERR_BAD_AUTH_SCHEME;
	}
    }
  if (pauth)
    *pauth = folder->authority;
  return status;
}

#else
#include <stdio.h>
#include <registrar0.h>
mu_record_t mu_pop_record = NULL;
mu_record_t mu_pops_record = NULL;
#endif /* ENABLE_POP */
