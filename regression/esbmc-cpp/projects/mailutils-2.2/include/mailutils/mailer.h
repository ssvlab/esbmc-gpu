/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2008, 2010 Free Software
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

#ifndef _MAILUTILS_MAILER_H
#define _MAILUTILS_MAILER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* During protocol traces, the data as well as headers will be dumped. */
#define MAILER_FLAG_DEBUG_DATA 0x0001

/* A url of NULL will cause the default to be used. */
extern int mu_mailer_create         (mu_mailer_t *, const char *url);
extern int mu_mailer_create_from_url (mu_mailer_t *pmailer, mu_url_t url);
  
extern void mu_mailer_destroy       (mu_mailer_t *);
extern int mu_mailer_open           (mu_mailer_t, int flags);
extern int mu_mailer_close          (mu_mailer_t);
extern int mu_mailer_send_message   (mu_mailer_t, mu_message_t,
				     mu_address_t from, mu_address_t to);

struct timeval;

extern int mu_mailer_send_fragments (mu_mailer_t mailer, mu_message_t msg,
				     size_t fragsize, struct timeval *delay,
				     mu_address_t from, mu_address_t to);

/* Called to set or get the default mailer url. */
extern int mu_mailer_set_url_default       (const char* url);
extern int mu_mailer_get_url_default       (const char** url);

/* Accessor functions. */
extern int mu_mailer_get_property   (mu_mailer_t, mu_property_t *);
extern int mu_mailer_get_stream     (mu_mailer_t, mu_stream_t *);
extern int mu_mailer_set_stream     (mu_mailer_t, mu_stream_t);
extern int mu_mailer_get_debug      (mu_mailer_t, mu_debug_t *);
extern int mu_mailer_set_debug      (mu_mailer_t, mu_debug_t);
extern int mu_mailer_get_observable (mu_mailer_t, mu_observable_t *);
extern int mu_mailer_get_url        (mu_mailer_t, mu_url_t *);

/* Utility functions, primarily for use of implementing concrete mailers. */

/* A valid from mu_address_t contains a single address that has a qualified
   email address. */
extern int mu_mailer_check_from     (mu_address_t from);
/* A valid to mu_address_t contains 1 or more addresses, that are
   qualified email addresses. */
extern int mu_mailer_check_to       (mu_address_t to);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_MAILER_H */
