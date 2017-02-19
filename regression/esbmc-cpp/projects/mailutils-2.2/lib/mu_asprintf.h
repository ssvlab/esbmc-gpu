/*
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU Library General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Library General Public License for more details.

   You should have received a copy of the GNU Library General
   Public License along with GNU Mailutils; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifndef MUASPRINTF_H
#define MUASPRINTF_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <stdarg.h>
#include <stdio.h>

#if !HAVE_DECL_VASPRINTF
extern int vasprintf (char **result, const char *format, va_list args);
#endif
#if !HAVE_DECL_ASPRINTF
#if __STDC__
extern int asprintf (char **result, const char *format, ...);
#else
extern int asprintf ();
#endif
#endif

#endif

