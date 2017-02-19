/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2005, 2007, 2008, 2010 Free Software
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

#ifndef _MAILUTILS_LIST_H
#define _MAILUTILS_LIST_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int mu_list_create   (mu_list_t *);
extern void mu_list_destroy (mu_list_t *);
extern int mu_list_append   (mu_list_t, void *item);
extern int mu_list_prepend  (mu_list_t, void *item);
extern int mu_list_insert   (mu_list_t list, void *item, void *new_item, 
                          int insert_before);
extern int mu_list_is_empty (mu_list_t);
extern int mu_list_count    (mu_list_t, size_t *pcount);
extern int mu_list_remove   (mu_list_t, void *item);
extern int mu_list_remove_nd  (mu_list_t, void *item);
extern int mu_list_replace  (mu_list_t list, void *old_item, void *new_item);  
extern int mu_list_replace_nd (mu_list_t list, void *old_item, void *new_item);  
extern int mu_list_get      (mu_list_t, size_t _index, void **pitem);
extern int mu_list_to_array (mu_list_t list, void **array, size_t count, size_t *pcount);
extern int mu_list_locate   (mu_list_t list, void *item, void **ret_item);
extern int mu_list_get_iterator (mu_list_t, mu_iterator_t *);

typedef int mu_list_action_t (void *item, void *cbdata);

extern int mu_list_do       (mu_list_t list, mu_list_action_t *action, void *cbdata);

typedef int (*mu_list_comparator_t) (const void*, const void*);

extern int _mu_list_ptr_comparator (const void*, const void*);
  
extern mu_list_comparator_t mu_list_set_comparator (mu_list_t,
						    mu_list_comparator_t);
extern int mu_list_get_comparator (mu_list_t, mu_list_comparator_t *);

extern void mu_list_free_item (void *item);

typedef void (*mu_list_destroy_item_t) (void *);
  
extern mu_list_destroy_item_t mu_list_set_destroy_item
              (mu_list_t list, mu_list_destroy_item_t destroy_item);


extern int mu_list_intersect_dup (mu_list_t *, mu_list_t, mu_list_t,
				  int (*dup_item) (void **, void *, void *),
				  void *);
extern int mu_list_intersect (mu_list_t *, mu_list_t, mu_list_t);  

extern int mu_list_insert_list (mu_list_t list, void *item, mu_list_t new_list,
				int insert_before);
extern void mu_list_append_list (mu_list_t list, mu_list_t new_list);
extern void mu_list_prepend_list (mu_list_t list, mu_list_t new_list);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_LIST_H */
