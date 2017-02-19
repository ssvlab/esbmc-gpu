/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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

#ifndef _MAILBOX0_H
#define _MAILBOX0_H

#ifdef DMALLOC
#  include <dmalloc.h>
#endif

#include <sys/types.h>
#include <stdio.h>

#include <mailutils/monitor.h>
#include <mailutils/mailbox.h>
#include <mailutils/iterator.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _mu_mailbox
{
  /* Data */
  mu_observable_t observable;
  mu_debug_t debug;
  mu_property_t property;
  mu_locker_t locker;
  mu_stream_t stream;
  mu_url_t url;
  int flags;
  mu_folder_t folder;
  mu_monitor_t monitor;
  mu_iterator_t iterator;
  
  /* Back pointer to the specific mailbox */
  void *data;

  /* Public methods */

  void (*_destroy)         (mu_mailbox_t);

  int  (*_open)            (mu_mailbox_t, int);
  int  (*_close)           (mu_mailbox_t);
  
  /* messages */
  int  (*_get_message)     (mu_mailbox_t, size_t, mu_message_t *);
  int  (*_append_message)  (mu_mailbox_t, mu_message_t);
  int  (*_messages_count)  (mu_mailbox_t, size_t *);
  int  (*_messages_recent) (mu_mailbox_t, size_t *);
  int  (*_message_unseen)  (mu_mailbox_t, size_t *);
  int  (*_expunge)         (mu_mailbox_t);
  int  (*_sync)            (mu_mailbox_t);
  int  (*_uidvalidity)     (mu_mailbox_t, unsigned long *);
  int  (*_uidnext)         (mu_mailbox_t, size_t *);
  int  (*_get_property)    (mu_mailbox_t, mu_property_t *);

  int  (*_scan)            (mu_mailbox_t, size_t, size_t *);
  int  (*_is_updated)      (mu_mailbox_t);

  int  (*_get_size)        (mu_mailbox_t, mu_off_t *);

  int  (*_quick_get_message) (mu_mailbox_t, mu_message_qid_t, mu_message_t *);
  int  (*_get_uidls) (mu_mailbox_t, mu_list_t);
};

#ifdef __cplusplus
}
#endif

#endif /* _MAILBOX0_H */
