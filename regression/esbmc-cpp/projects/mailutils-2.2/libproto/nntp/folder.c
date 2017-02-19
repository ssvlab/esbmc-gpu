/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifdef ENABLE_NNTP

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/nntp.h>
#include <mailutils/errno.h>
#include <mailutils/mailbox.h>
#include <mailutils/registrar.h>
#include <mailutils/url.h>

#include <folder0.h>
#include "nntp0.h"

/* We export url parsing and the initialisation of
   the mailbox, via the register entry/record.  */

static struct _mu_record _nntp_record =
{
  MU_NNTP_PRIO,
  MU_NNTP_URL_SCHEME,
  _nntp_url_init, /* Url init.  */
  _nntp_mailbox_init, /* Mailbox init.  */
  NULL, /* Mailer init.  */
  _nntp_folder_init, /* Folder init.  */
  NULL, /* No need for an back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};
mu_record_t mu_nntp_record = &_nntp_record;

static int  nntp_folder_open    (mu_folder_t, int);
static int  nntp_folder_close   (mu_folder_t);
static void nntp_folder_destroy (mu_folder_t folder);
static int  nntp_folder_list    (mu_folder_t folder, const char *ref,
				 void *name, int flags,
				 size_t max,
				 mu_list_t flist,
				 mu_folder_enumerate_fp efp, void *edp);

int
_nntp_folder_init (mu_folder_t folder)
{
  f_nntp_t f_nntp;

  f_nntp = folder->data = calloc (1, sizeof (*f_nntp));
  if (f_nntp == NULL)
    return ENOMEM;

  f_nntp->folder = folder;

  folder->_destroy = nntp_folder_destroy;

  folder->_open = nntp_folder_open;
  folder->_close = nntp_folder_close;

  folder->_list = nntp_folder_list;
  /* Not supported.
    folder->_lsub = folder_nntp_lsub;
    folder->_subscribe = folder_nntp_subscribe;
    folder->_unsubscribe = folder_nntp_unsubscribe;
    folder->_delete = folder_nntp_delete;
    folder->_rename = folder_nntp_rename;
  */

  return 0;
}

static int
nntp_folder_open (mu_folder_t folder, int flags)
{
  f_nntp_t f_nntp = folder->data;
  mu_stream_t carrier = NULL;
  const char *host;
  long port = MU_NNTP_DEFAULT_PORT; /* default nntp port.  */
  int status = 0;

  /* If we are already open for business, noop.  */
  mu_monitor_wrlock (folder->monitor);
  if (f_nntp->isopen)
    {
      mu_monitor_unlock (folder->monitor);
      return 0;
    }
  mu_monitor_unlock (folder->monitor);

  /* Fetch the server name and the port in the mu_url_t.  */
  status = mu_url_sget_host (folder->url, &host);
  if (status != 0)
    return status;
  mu_url_get_port (folder->url, &port);

  folder->flags = flags;

  /* Create the networking stack.  */
  status = mu_tcp_stream_create (&carrier, host, port, folder->flags);
  if (status != 0)
    return status;
  /* Ask for the stream internal buffering mechanism scheme.  */
  mu_stream_setbufsiz (carrier, BUFSIZ);
  MU_DEBUG2 (folder->debug, MU_DEBUG_PROT, "folder_nntp_open (%s:%ld)\n", 
             host, port);

  status = mu_nntp_create (&f_nntp->nntp);
  if (status == 0)
    {
      status = mu_nntp_set_carrier (f_nntp->nntp, carrier);
      if (status == 0)
	{
	  status = mu_nntp_connect (f_nntp->nntp);
	  if (status == 0)
	    {
	      mu_monitor_wrlock (folder->monitor);
	      f_nntp->isopen++;
	      mu_monitor_unlock (folder->monitor);
	    }
	}
    }

  return status;
}

static int
nntp_folder_close (mu_folder_t folder)
{
  f_nntp_t f_nntp = folder->data;
  int status = 0;

  mu_monitor_wrlock (folder->monitor);
  f_nntp->isopen--;
  if (f_nntp->isopen)
    {
      mu_monitor_unlock (folder->monitor);
      return 0;
    }
  mu_monitor_unlock (folder->monitor);
  status = mu_nntp_quit (f_nntp->nntp);
  f_nntp->selected = NULL;
  return status;

}

/* Destroy the folder resources.  */
static void
nntp_folder_destroy (mu_folder_t folder)
{
  if (folder->data)
    {
      f_nntp_t f_nntp = folder->data;
      if (f_nntp->nntp)
        mu_nntp_destroy (&f_nntp->nntp);
      free (f_nntp);
      folder->data = NULL;
    }
}


static int
nntp_folder_list (mu_folder_t folder, const char *ref, void *pat, int flags,
		  size_t max_level, mu_list_t flist,
		  mu_folder_enumerate_fp efp, void *edp)
{
  return ENOTSUP;
}
#else
#include <stdio.h>
#include <registrar0.h>
mu_record_t mu_nntp_record = NULL;
#endif
