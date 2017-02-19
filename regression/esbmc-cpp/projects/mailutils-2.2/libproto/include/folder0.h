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

#ifndef _FOLDER0_H
#define _FOLDER0_H

#ifdef DMALLOC
#  include <dmalloc.h>
#endif

#include <sys/types.h>
#include <stdio.h>

#include <mailutils/monitor.h>
#include <mailutils/folder.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MU_FOLDER_LIST 0
#define MU_FOLDER_ENUM 1
  
struct _mu_folder
{
  /* Data */
  mu_authority_t authority;
  mu_observable_t observable;
  mu_debug_t debug;
  mu_stream_t stream;
  mu_monitor_t monitor;
  mu_url_t url;
  int flags;
  int ref;
  size_t uid;

  /* Back pointer to the specific mailbox */
  void *data;

  /* Public methods */

  void (*_destroy)     (mu_folder_t);

  int  (*_open)        (mu_folder_t, int flag);
  int  (*_close)       (mu_folder_t);
  int  (*_list)        (mu_folder_t, const char *, void *, int, size_t,
			mu_list_t, mu_folder_enumerate_fp, void *);
  int  (*_lsub)        (mu_folder_t, const char *, const char *,
		        mu_list_t);
  mu_folder_match_fp   _match;
  int  (*_delete)      (mu_folder_t, const char *);
  int  (*_rename)      (mu_folder_t, const char *, const char *);
  int  (*_subscribe)   (mu_folder_t, const char *);
  int  (*_unsubscribe) (mu_folder_t, const char *);
};

#ifdef __cplusplus
}
#endif

#endif /* _FOLDER0_H */
