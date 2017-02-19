/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2005, 2006, 2007, 2010 Free Software Foundation,
   Inc.

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

#ifndef _MAILUTILS_NLS_H
#define _MAILUTILS_NLS_H

#include <mailutils/types.h>
#if defined(DEFAULT_TEXT_DOMAIN)
# include <locale.h>
# include <gettext.h>
# define _(string) gettext (string)
# define N_(string) string
# ifdef ENABLE_NLS
#  define MU_APP_INIT_NLS()                                    \
    do                                                         \
      {                                                        \
        setlocale (LC_ALL, "");                                \
        mu_init_nls ();                                        \
      }                                                        \
    while (0)
# else
#  define MU_APP_INIT_NLS()
# endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern void mu_init_nls (void);
extern char *mu_set_locale (const char *locale);
void mu_restore_locale (void);

#ifdef __cplusplus
}
#endif

#endif
