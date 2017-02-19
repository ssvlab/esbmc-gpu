/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2008, 2009, 2010 Free
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

#ifndef _MAILUTILS_URL_H
#define _MAILUTILS_URL_H	1

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int  mu_url_create    (mu_url_t *, const char *name);
extern int  mu_url_dup       (mu_url_t old_url, mu_url_t *new_url);
extern int  mu_url_uplevel   (mu_url_t url, mu_url_t *upurl);

extern void mu_url_destroy   (mu_url_t *);
extern int  mu_url_parse     (mu_url_t);

extern int mu_url_sget_scheme  (const mu_url_t, const char **);
extern int mu_url_aget_scheme  (const mu_url_t, char **);  
extern int mu_url_get_scheme  (const mu_url_t, char *, size_t, size_t *);
  
extern int mu_url_sget_user  (const mu_url_t, const char **);
extern int mu_url_aget_user  (const mu_url_t, char **);  
extern int mu_url_get_user  (const mu_url_t, char *, size_t, size_t *);

extern int mu_url_get_secret (const mu_url_t url, mu_secret_t *psecret);
  
extern int mu_url_sget_auth  (const mu_url_t, const char **);
extern int mu_url_aget_auth  (const mu_url_t, char **);  
extern int mu_url_get_auth  (const mu_url_t, char *, size_t, size_t *);

extern int mu_url_sget_host  (const mu_url_t, const char **);
extern int mu_url_aget_host  (const mu_url_t, char **);  
extern int mu_url_get_host  (const mu_url_t, char *, size_t, size_t *);
  
extern int mu_url_sget_path  (const mu_url_t, const char **);
extern int mu_url_aget_path  (const mu_url_t, char **);  
extern int mu_url_get_path  (const mu_url_t, char *, size_t, size_t *);

extern int mu_url_sget_query (const mu_url_t url, size_t *qc, char ***qv);
extern int mu_url_aget_query (const mu_url_t url, size_t *qc, char ***qv);
  
extern int mu_url_get_port    (const mu_url_t, long *);

int mu_url_sget_fvpairs (const mu_url_t url, size_t *fvc, char ***fvp);
int mu_url_aget_fvpairs (const mu_url_t url, size_t *pfvc, char ***pfvp);

extern int mu_url_expand_path (mu_url_t url);
extern const char *mu_url_to_string   (const mu_url_t);

extern int mu_url_set_scheme (mu_url_t url, const char *scheme);
  
extern int mu_url_is_scheme   (mu_url_t, const char *scheme);

extern int mu_url_is_same_scheme (mu_url_t, mu_url_t);
extern int mu_url_is_same_user   (mu_url_t, mu_url_t);
extern int mu_url_is_same_path   (mu_url_t, mu_url_t);
extern int mu_url_is_same_host   (mu_url_t, mu_url_t);
extern int mu_url_is_same_port   (mu_url_t, mu_url_t);

extern char *mu_url_decode_len (const char *s, size_t len);  
extern char *mu_url_decode     (const char *s);

extern int mu_url_is_ticket   (mu_url_t ticket, mu_url_t url);
extern int mu_url_init (mu_url_t url, int port, const char *scheme);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_URL_H */
