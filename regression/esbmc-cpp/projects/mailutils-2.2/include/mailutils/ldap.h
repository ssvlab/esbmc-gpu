/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifndef _MAILUTILS_LDAP_H
#define _MAILUTILS_LDAP_H

#ifdef __cplusplus
extern "C" {
#endif

struct mu_ldap_module_config
{
  int enable;
  char *url;
  char *base;
  char *binddn;
  char *passwd;
  int tls;
  /* FIXME: sasl */
  int debug;
  char *getpwnam_filter;
  char *getpwuid_filter;
  mu_assoc_t field_map;
};

#ifdef __cplusplus
}
#endif

#endif
