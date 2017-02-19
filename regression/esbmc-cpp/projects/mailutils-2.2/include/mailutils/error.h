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

#ifndef _MAILUTILS_ERROR_H
#define _MAILUTILS_ERROR_H

#include <stdarg.h>

#include <mailutils/types.h>
#include <mailutils/diag.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*mu_error_pfn_t) (const char *fmt, va_list ap);

extern int mu_verror (const char *fmt, va_list ap);
extern int mu_error (const char *fmt, ...) MU_PRINTFLIKE(1,2);
extern void mu_error_set_print (mu_error_pfn_t) __attribute__((deprecated));

int mu_default_error_printer (const char *fmt, va_list ap);
int mu_syslog_error_printer (const char *fmt, va_list ap);
  
#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_ERROR_H */
