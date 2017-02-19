/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <mailutils/kwd.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>

int
mu_kwd_xlat_name_len (mu_kwd_t *kwtab, const char *str, size_t len, int *pres)
{
  for (; kwtab->name; kwtab++)
    {
      size_t kwlen = strlen (kwtab->name);
      if (kwlen == len && memcmp (kwtab->name, str, len) == 0)
	{
	  *pres = kwtab->tok;
	  return 0;
	}
    }
  return MU_ERR_NOENT;
}

int
mu_kwd_xlat_name_len_ci (mu_kwd_t *kwtab, const char *str, size_t len,
			 int *pres)
{
  for (; kwtab->name; kwtab++)
    {
      size_t kwlen = strlen (kwtab->name);
      if (kwlen == len && mu_c_strncasecmp (kwtab->name, str, len) == 0)
	{
	  *pres = kwtab->tok;
	  return 0;
	}
    }
  return MU_ERR_NOENT;
}

int
mu_kwd_xlat_name (mu_kwd_t *kwtab, const char *str, int *pres)
{
  for (; kwtab->name; kwtab++)
    if (strcmp (kwtab->name, str) == 0)
      {
	*pres = kwtab->tok;
	return 0;
      }
  return MU_ERR_NOENT;
}

int
mu_kwd_xlat_name_ci (mu_kwd_t *kwtab, const char *str, int *pres)
{
  for (; kwtab->name; kwtab++)
    if (mu_c_strcasecmp (kwtab->name, str) == 0)
      {
	*pres = kwtab->tok;
	return 0;
      }
  return MU_ERR_NOENT;
}
     

int
mu_kwd_xlat_tok (mu_kwd_t *kwtab, int tok, const char **pres)
{
  for (; kwtab->name; kwtab++)
    if (kwtab->tok == tok)
      {
	*pres = kwtab->name;
	return 0;
      }
  return MU_ERR_NOENT;
}  
