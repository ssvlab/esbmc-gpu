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

#ifndef _MAILUTILS_PROPERTY_H
#define _MAILUTILS_PROPERTY_H

#include <sys/types.h>

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int  mu_property_create   (mu_property_t *, void *);
extern void mu_property_destroy (mu_property_t *, void *);
extern void *mu_property_get_owner (mu_property_t);

extern int  mu_property_set_value (mu_property_t, const char *, const char *, int);
extern int mu_property_get_value (mu_property_t, const char *, char *, size_t, 
				  size_t *);
extern int mu_property_sget_value (mu_property_t prop, const char *key,
				   const char **buffer);
extern int mu_property_aget_value (mu_property_t prop, const char *key,
				   char **buffer);

/* Helper functions.  */
extern int mu_property_set  (mu_property_t, const char *);
extern int mu_property_unset (mu_property_t, const char *);
extern int mu_property_is_set (mu_property_t, const char *);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_PROPERTY_H */
