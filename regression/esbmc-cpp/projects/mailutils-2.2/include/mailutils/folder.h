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

#ifndef _MAILUTILS_FOLDER_H
# define _MAILUTILS_FOLDER_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mu_list_response
{
  int type;             /* MU_FOLDER_ATTRIBUTE_.* flags */
  int level;
  int separator;
  char *name;
};

typedef int (*mu_folder_match_fp) (const char *, void *, int);
typedef int (*mu_folder_enumerate_fp) (mu_folder_t, struct mu_list_response *,
				       void *data);
  
/* Constructor/destructor and possible types.  */
extern int  mu_folder_create         (mu_folder_t *, const char *);
extern int  mu_folder_create_from_record (mu_folder_t *, mu_url_t url,
					  mu_record_t);
  
extern void mu_folder_destroy        (mu_folder_t *);

extern int  mu_folder_open           (mu_folder_t, int flag);
extern int  mu_folder_close          (mu_folder_t);

extern int  mu_folder_delete         (mu_folder_t, const char *);
extern int  mu_folder_rename         (mu_folder_t, const char *, const char *);
extern int  mu_folder_subscribe      (mu_folder_t, const char *);
extern int  mu_folder_unsubscribe    (mu_folder_t, const char *);
extern int  mu_folder_list           (mu_folder_t, const char *, void *,
				      size_t, mu_list_t *);
extern int  mu_folder_enumerate      (mu_folder_t, const char *,
				      void *, int, 
				      size_t, mu_list_t *,
				      mu_folder_enumerate_fp, void *);
extern int  mu_folder_lsub           (mu_folder_t, const char *, const char *,
				      mu_list_t *);

/* Stream settings.  */
extern int  mu_folder_get_stream     (mu_folder_t, mu_stream_t *);
extern int  mu_folder_set_stream     (mu_folder_t, mu_stream_t);

  /* Match function */
extern int mu_folder_set_match (mu_folder_t folder, mu_folder_match_fp pmatch);
extern int mu_folder_get_match (mu_folder_t folder,
				mu_folder_match_fp *pmatch);
  
  /* Notifications.  */
extern int  mu_folder_get_observable (mu_folder_t, mu_observable_t *);

  /* Debug.  */
extern int  mu_folder_has_debug      (mu_folder_t folder);
extern int  mu_folder_get_debug      (mu_folder_t, mu_debug_t *);
extern int  mu_folder_set_debug      (mu_folder_t, mu_debug_t);

/* Authentication.  */
extern int  mu_folder_get_authority  (mu_folder_t, mu_authority_t *);
extern int  mu_folder_set_authority  (mu_folder_t, mu_authority_t);

/* URL.  */
extern int  mu_folder_get_url        (mu_folder_t, mu_url_t *);
extern int  mu_folder_set_url        (mu_folder_t, mu_url_t);

/* FIXME: not implemented */
extern int  mu_folder_decrement      (mu_folder_t);

extern void mu_list_response_free    (void *data);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_FOLDER_H */
