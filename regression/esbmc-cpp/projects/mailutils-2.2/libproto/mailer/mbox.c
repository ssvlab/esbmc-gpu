/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

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
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <mailutils/address.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/property.h>
#include <mailutils/mailer.h>
#include <mailutils/url.h>
#include <mailutils/mutil.h>
#include <mailbox0.h>
#include <mailer0.h>

struct remote_mbox_data
{
  mu_mailer_t mailer;
};

static void
remote_mbox_destroy (mu_mailbox_t mailbox)
{
  if (mailbox->data)
    {
      struct remote_mbox_data *dat = mailbox->data;
      mu_mailer_destroy (&dat->mailer);
      free (dat);
      mailbox->data = NULL;
    }
}

static int
remote_mbox_open (mu_mailbox_t mbox, int flags)
{
  struct remote_mbox_data *dat = mbox->data;
  int status;
  int mflags = 0;
  mu_log_level_t lev = 0;
  
  if (!dat->mailer)
    return EINVAL;

  mu_debug_get_level (mbox->debug, &lev);
  if (lev & MU_DEBUG_TRACE7)
    mflags = MAILER_FLAG_DEBUG_DATA;
  status = mu_mailer_open (dat->mailer, mflags);
  if (status)
    {
      MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR,
		 "cannot open mailer: %s\n", mu_strerror (status));
      return status;
    }
  if (lev & MU_DEBUG_INHERIT)
    {
      mu_debug_t debug;
      if (mu_mailer_get_debug (dat->mailer, &debug) == 0)
	mu_debug_set_level (debug, lev);
    }
  mbox->flags = flags;
  return 0;
}

static int
remote_mbox_close (mu_mailbox_t mbox)
{
  struct remote_mbox_data *dat = mbox->data;
  int status;
  
  MU_DEBUG (mbox->debug, MU_DEBUG_TRACE1, "remote_mbox_close\n");
  status = mu_mailer_close (dat->mailer);
  if (status)
    MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR, "closing mailer failed: %s\n",
	       mu_strerror (status));
  return status;
}

static int
mkaddr (mu_mailbox_t mbox, mu_property_t property,
	const char *key, mu_address_t *addr)
{
  const char *str = NULL;
  mu_property_sget_value (property, key, &str);
  if (str && *str)
    {
      int status = mu_address_create (addr, str);
      if (status)
	{
	  MU_DEBUG3 (mbox->debug, MU_DEBUG_ERROR,
		     "%s: %s mu_address_create failed: %s\n",
		     str, key, mu_strerror (status));
	  return status;
	}
    }
  else
    *addr = NULL;
  return 0;
}


static int
remote_mbox_append_message (mu_mailbox_t mbox, mu_message_t msg)
{
  struct remote_mbox_data *dat = mbox->data;
  int status;
  mu_property_t property = NULL;
  mu_address_t from, to;
  
  if (!dat->mailer)
    return EINVAL;

  status = mu_mailbox_get_property (mbox, &property);
  if (status)
    MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR, "failed to get property: %s\n",
	       mu_strerror (status));

  mkaddr (mbox, property, "FROM", &from);
  mkaddr (mbox, property, "TO", &to);
  if (!to)
    {
      const char *rcpt;
      
      status = mu_url_sget_user (mbox->url, &rcpt);
      if (status != MU_ERR_NOENT)
	{
	  const char *host;
	  struct mu_address hint;
	  
	  if (status)
	    {
	      MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR,
			 "failed to get recipient from the url: %s\n",
			 mu_strerror (status));
	      return status;
	    }

	  mu_url_sget_host (mbox->url, &host);
	  hint.domain = (char*) host;
	  status = mu_address_create_hint (&to, rcpt, &hint, 
	                                   MU_ADDR_HINT_DOMAIN);
      
	  if (status)
	    {
	      MU_DEBUG3 (mbox->debug, MU_DEBUG_ERROR,
			 "%s: %s mu_address_create failed: %s\n",
			 rcpt, "TO", mu_strerror (status));
	      return status;
	    }
	}
    }
  
  status = mu_mailer_send_message (dat->mailer, msg, from, to);

  if (status)
    MU_DEBUG1 (mbox->debug, MU_DEBUG_ERROR,
	       "Sending message failed: %s\n", mu_strerror (status));
  return status;
}

static int
remote_mbox_scan (mu_mailbox_t mbox, size_t offset, size_t *pcount)
{
  if (pcount)
    *pcount = 0;
  return 0;
}

static int
remote_get_size (mu_mailbox_t mbox, mu_off_t *psize)
{
  if (psize)
    *psize = 0;
  return 0;
}

static int
remote_sync (mu_mailbox_t mbox)
{
  return 0;
}

int
_mu_mailer_mailbox_init (mu_mailbox_t mailbox)
{
  struct remote_mbox_data *dat;
  int rc;
  mu_mailer_t mailer;
  mu_url_t url;
  
  if (mailbox == NULL)
    return EINVAL;

  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1,
	     "_mu_mailer_mailbox_init(%s)\n",
	     mu_url_to_string (mailbox->url));

  rc = mu_url_dup (mailbox->url, &url);
  if (rc)
    return rc;

  rc = mu_mailer_create_from_url (&mailer, url);
  if (rc)
    {
      MU_DEBUG2 (mailbox->debug, MU_DEBUG_ERROR,
		 "_mu_mailer_mailbox_init(%s): cannot create mailer: %s\n",
		 mu_url_to_string (url), mu_strerror (rc));
      mu_url_destroy (&url);
      return rc;
    }
  
  dat = mailbox->data = calloc (1, sizeof (*dat));
  if (dat == NULL)
    {
      mu_mailer_destroy (&mailer);
      return ENOMEM;
    }
  dat->mailer = mailer;

  mailbox->_destroy = remote_mbox_destroy;
  mailbox->_open = remote_mbox_open;
  mailbox->_close = remote_mbox_close;
  mailbox->_append_message = remote_mbox_append_message;
  mailbox->_scan = remote_mbox_scan;
  mailbox->_get_size = remote_get_size;
  mailbox->_sync = remote_sync;

  return 0;
}

int
_mu_mailer_folder_init (mu_folder_t folder MU_ARG_UNUSED)
{
  return 0;
}
