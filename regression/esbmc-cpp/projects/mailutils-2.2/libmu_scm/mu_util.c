/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2006, 2007, 2010 Free Software
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

#include "mu_scm.h"

SCM_DEFINE_PUBLIC (scm_mu_getpwuid, "mu-getpwuid", 1, 0, 0,
		   (SCM user),
"Look up an entry in the user database. @var{User} can be an integer,\n"
"or a string, giving the behaviour of @code{mu_get_auth_by_uid} or\n"
"@code{mu_get_auth_by_name} respectively.\n"
"\n"
"Returns a vector with fields corresponding to those of the @code{mu_auth_data}\n"
"entry in question. If no matching entry was found, returns @code{#f}.\n")
#define FUNC_NAME s_scm_mu_getpwuid
{
  SCM result;
  struct mu_auth_data *entry;
  SCM *ve;
  scm_t_array_handle handle;
  
  result = scm_c_make_vector (8, SCM_UNSPECIFIED);
  ve = scm_vector_writable_elements (result,
				     &handle,
				     NULL, NULL);
  
  if (scm_is_integer (user))
    {
      entry = mu_get_auth_by_uid (scm_to_int (user));
    }
  else
    {
      char *s;
      
      SCM_VALIDATE_STRING (1, user);
      s = scm_to_locale_string (user);
      entry = mu_get_auth_by_name (s);
      free (s);
    }
  if (!entry)
    mu_scm_error (FUNC_NAME, errno,
		  "Cannot get user credentials", SCM_BOOL_F);

  ve[0] = scm_from_locale_string (entry->name);
  ve[1] = scm_from_locale_string (entry->passwd);
  ve[2] = scm_from_ulong ((unsigned long) entry->uid);
  ve[3] = scm_from_ulong ((unsigned long) entry->gid);
  ve[4] = scm_from_locale_string (entry->gecos);
  ve[5] = scm_from_locale_string (entry->dir ? entry->dir : "");
  ve[6] = scm_from_locale_string (entry->shell ? entry->shell : "");
  ve[7] = scm_from_locale_string (entry->mailbox);

  scm_array_handle_release (&handle);
  
  mu_auth_data_free (entry);
  return result;
}
#undef FUNC_NAME

void
mu_scm_mutil_init ()
{
#include "mu_util.x"
}
