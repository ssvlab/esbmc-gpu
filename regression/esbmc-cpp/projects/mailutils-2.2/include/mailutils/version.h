/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2008, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_VERSION_H
#define _MAILUTILS_VERSION_H

#include <stdio.h>  

#ifdef __cplusplus
extern "C" {
#endif

struct mu_conf_option
{
  char *name;
  char *descr;
};
  
extern char *mu_license_text;
extern void mu_print_options (void);
extern void mu_fprint_options (FILE *fp, int verbose);
extern void mu_fprint_conf_option (FILE *fp, const struct mu_conf_option *opt,
				   int verbose);
extern const struct mu_conf_option *mu_check_option (char *name);

#ifdef __cplusplus
}
#endif

#endif
