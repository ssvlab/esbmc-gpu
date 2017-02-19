/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2002, 2006, 2007, 2008, 2010 Free Software Foundation,
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

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <mailutils/nls.h>
#include <locale.h>

/* Initialize Native Language Support */

char *mu_locale_set;

#ifndef HAVE_SETLOCALE
# define setlocale(c,s) NULL
#endif

/* Set locale via LC_ALL.  */
char *
mu_set_locale (const char *locale)
{
#if defined HAVE_SETLOCALE
  return setlocale (LC_ALL, locale);
#else
  return NULL;
#endif
}

void
mu_restore_locale (void)
{
  if (mu_locale_set)
    mu_set_locale (mu_locale_set);
}

void
mu_init_nls (void)
{
#ifdef ENABLE_NLS
  mu_locale_set = mu_set_locale ("");
  bindtextdomain (PACKAGE, LOCALEDIR);
#endif /* ENABLE_NLS */
}

