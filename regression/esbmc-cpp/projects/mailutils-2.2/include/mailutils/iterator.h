/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2004, 2005, 2007, 2010 Free Software
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

#ifndef _MAILUTILS_ITERATOR_H
#define _MAILUTILS_ITERATOR_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mu_itrctl_req
  {
    mu_itrctl_tell,          /* Return current position in the object */
    mu_itrctl_delete,        /* Delete current element */
    mu_itrctl_delete_nd,     /* Delete current element, non-destructive */
    mu_itrctl_replace,       /* Replace current element */
    mu_itrctl_replace_nd,    /* Replace current element, non-destructive */
    mu_itrctl_insert,        /* Insert new element in the current position */
    mu_itrctl_insert_list,   /* Insert a list of elements */
    mu_itrctl_qry_direction, /* Query iteration direction */
    mu_itrctl_set_direction  /* Set iteration direction */
  };
  
extern int mu_iterator_create   (mu_iterator_t *, void *);
extern int mu_iterator_dup      (mu_iterator_t *piterator, mu_iterator_t orig);
extern void mu_iterator_destroy (mu_iterator_t *);
extern int mu_iterator_first    (mu_iterator_t);
extern int mu_iterator_next     (mu_iterator_t);
extern int mu_iterator_skip (mu_iterator_t iterator, ssize_t count);
extern int mu_iterator_current  (mu_iterator_t, void **pitem);
extern int mu_iterator_current_kv (mu_iterator_t,
				   const void **key, void **pitem);  
extern int mu_iterator_is_done  (mu_iterator_t);

extern int mu_iterator_ctl (mu_iterator_t, enum mu_itrctl_req, void *);
  
extern int mu_iterator_attach (mu_iterator_t *root, mu_iterator_t iterator);
extern int mu_iterator_detach (mu_iterator_t *root, mu_iterator_t iterator);
extern void mu_iterator_advance (mu_iterator_t iterator, void *e);
  
extern int mu_iterator_set_first (mu_iterator_t, int (*first) (void *));  
extern int mu_iterator_set_next (mu_iterator_t, int (*next) (void *));  
extern int mu_iterator_set_getitem (mu_iterator_t,
				 int (*getitem) (void *, void **,
						 const void **pkey));  
extern int mu_iterator_set_finished_p (mu_iterator_t,
				    int (*finished_p) (void *));  
extern int mu_iterator_set_dup (mu_iterator_t itr,
			     int (*dup) (void **ptr, void *data));
extern int mu_iterator_set_destroy (mu_iterator_t itr,
				 int (*destroy) (mu_iterator_t, void *data));
extern int mu_iterator_set_curitem_p (mu_iterator_t itr,
				   int (*curitem_p) (void *, void *));
extern int mu_iterator_set_itrctl (mu_iterator_t itr,
				   int (*itrctl) (void *,
						  enum mu_itrctl_req,
						  void *));
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_ITERATOR_H */
