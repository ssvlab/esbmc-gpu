/* GNU mailutils - a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2010 Free
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

#ifndef _MAILUTILS_REFCOUNT_H
#define _MAILUTILS_REFCOUNT_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int  mu_refcount_create  (mu_refcount_t *);
extern void mu_refcount_destroy (mu_refcount_t *);
extern unsigned mu_refcount_value (mu_refcount_t);
extern unsigned mu_refcount_inc   (mu_refcount_t);
extern unsigned mu_refcount_dec   (mu_refcount_t);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_REFCOUNT_H */
