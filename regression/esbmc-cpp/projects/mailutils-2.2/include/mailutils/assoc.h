/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_ASSOC_H
#define _MAILUTILS_ASSOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mailutils/types.h>

#define MU_ASSOC_COPY_KEY 0x01
#define MU_ASSOC_ICASE    0x02
  
typedef void (*mu_assoc_free_fn) (void *data);
  
int mu_assoc_create (mu_assoc_t *passoc, size_t elsize, int flags);
void mu_assoc_destroy (mu_assoc_t *passoc);
void mu_assoc_clear (mu_assoc_t assoc);  
void *mu_assoc_ref (mu_assoc_t assoc, const char *name);
int mu_assoc_install (mu_assoc_t assoc, const char *name, void *value);
int mu_assoc_ref_install (mu_assoc_t assoc, const char *name, void **pval);
int mu_assoc_get_iterator (mu_assoc_t assoc, mu_iterator_t *piterator);
int mu_assoc_remove_ref (mu_assoc_t assoc, void *data);
int mu_assoc_remove (mu_assoc_t assoc, const char *name);  
int mu_assoc_set_free (mu_assoc_t assoc, mu_assoc_free_fn fn);
int mu_assoc_count (mu_assoc_t assoc, size_t *pcount);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_ASSOC_H */

     
