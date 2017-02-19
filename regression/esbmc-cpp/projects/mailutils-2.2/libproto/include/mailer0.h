/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2009, 2010 Free Software
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

#ifndef _MAILER0_H
#define _MAILER0_H

#ifdef DMALLOC
#  include <dmalloc.h>
#endif

#include <sys/types.h>
#include <mailutils/mailer.h>
#include <mailutils/monitor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default mailer URL. */

#define MAILER_URL_DEFAULT "sendmail:"

#define MAILER_LINE_BUF_SIZE	1000

struct _mu_mailer
{
  mu_stream_t stream;
  mu_observable_t observable;
  mu_debug_t debug;
  mu_url_t url;
  int flags;
  mu_monitor_t monitor;
  mu_property_t property;

  /* Pointer to the specific mailer data.  */
  void *data;

  /* Public methods.  */
  void (*_destroy)     (mu_mailer_t);
  int (*_open)         (mu_mailer_t, int flags);
  int (*_close)        (mu_mailer_t);
  int (*_send_message) (mu_mailer_t, mu_message_t, mu_address_t, mu_address_t);
};

int _mu_mailer_mailbox_init (mu_mailbox_t mailbox);
int _mu_mailer_folder_init (mu_folder_t folder MU_ARG_UNUSED);
  
#define MAILER_NOTIFY(mailer, type) \
  if (mailer->observer) observer_notify (mailer->observer, type)

#ifdef __cplusplus
}
#endif

#endif /* MAILER0_H */
