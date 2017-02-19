/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_SECRET_H
#define _MAILUTILS_SECRET_H

#include <mailutils/types.h>

#ifdef __cplusplus
extern "C" {
#endif

int mu_secret_create (mu_secret_t *psec, const char *value, size_t len);
int mu_secret_dup (mu_secret_t sec, mu_secret_t *newsec);
void mu_secret_destroy (mu_secret_t *psec);
const char *mu_secret_password (mu_secret_t sec);
void mu_secret_password_unref (mu_secret_t sec);
size_t mu_secret_length (mu_secret_t sec);
void mu_secret_ref (mu_secret_t sec);
int mu_secret_unref (mu_secret_t sec);
  
#ifdef __cplusplus
}
#endif
  
#endif
