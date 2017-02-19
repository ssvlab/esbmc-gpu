/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2010 Free Software
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

typedef int (*address_get_fp) (mu_address_t, size_t, char **);

static SCM
_get_address_part (const char *func_name, address_get_fp fun,
		   SCM address, SCM num)
{
  mu_address_t addr;
  char *str;
  SCM ret;
  int n;
  int status;
  
  SCM_ASSERT (scm_is_string (address), address, SCM_ARG1, func_name);

  if (!SCM_UNBNDP (num))
    {
      SCM_ASSERT (scm_is_integer (num), num, SCM_ARG1, func_name);
      n = scm_to_int (num);
    }
  else
    n = 1;

  str = scm_to_locale_string (address);
  if (!str[0])
    {
      free (str);
      mu_scm_error (func_name, 0, "Empty address", SCM_BOOL_F);
    }
  
  status = mu_address_create (&addr, str);
  free (str);
  if (status)
    mu_scm_error (func_name, status, "Cannot create address", SCM_BOOL_F);

  status = (*fun) (addr, n, &str);
  mu_address_destroy (&addr);

  if (status == 0)
    ret = scm_from_locale_string (str);
  else
    {
      free (str);
      mu_scm_error (func_name, status,
		    "Underlying function failed", SCM_BOOL_F);
    }
  
  free (str);
  return ret;
}

SCM_DEFINE_PUBLIC (scm_mu_address_get_personal, "mu-address-get-personal", 1, 1, 0,
	    (SCM address, SCM num),
	    "Return personal part of the @var{num}th email address from @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_personal
{
  return _get_address_part (FUNC_NAME, 
			    mu_address_aget_personal, address, num);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_address_get_comments, "mu-address-get-comments", 1, 1, 0,
	    (SCM address, SCM num),
	    "Return comment part of the @var{num}th email address from @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_comments
{
  return _get_address_part (FUNC_NAME, 
			    mu_address_aget_comments, address, num);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_address_get_email, "mu-address-get-email", 1, 1, 0,
	    (SCM address, SCM num),
	    "Return email part of the @var{num}th email address from @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_email
{
  return _get_address_part (FUNC_NAME, 
			    mu_address_aget_email, address, num);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_address_get_domain, "mu-address-get-domain", 1, 1, 0,
	    (SCM address, SCM num),
	    "Return domain part of the @var{num}th email address from @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_domain
{
  return _get_address_part (FUNC_NAME, 
			    mu_address_aget_domain, address, num);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_address_get_local, "mu-address-get-local", 1, 1, 0,
	    (SCM address, SCM num),
	    "Return local part of the @var{num}th email address from @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_local
{
  return _get_address_part (FUNC_NAME, 
			    mu_address_aget_local_part, address, num);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_address_get_count, "mu-address-get-count", 1, 0, 0,
	    (SCM address),
	    "Return number of parts in email address @var{address}.\n")
#define FUNC_NAME s_scm_mu_address_get_count
{
  mu_address_t addr;
  size_t count = 0;
  int status;
  char *str;
  
  SCM_ASSERT (scm_is_string (address), address, SCM_ARG1, FUNC_NAME);

  str = scm_to_locale_string (address);
  status = mu_address_create (&addr, str);
  free (str);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot create address for ~A",
		  scm_list_1 (address));

  mu_address_get_count (addr, &count);
  mu_address_destroy (&addr);
  return scm_from_size_t (count);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_username_to_email, "mu-username->email", 0, 1, 0,
	    (SCM name),
"Deduce user's email address from his username. If @var{name} is omitted, \n"
"current username is assumed\n")
#define FUNC_NAME s_scm_mu_username_to_email
{
  char *username;
  char *email;
  SCM ret;
  
  if (SCM_UNBNDP (name))
    username = NULL;
  else
    {
      SCM_ASSERT (scm_is_string (name), name, SCM_ARG1, FUNC_NAME);
      username = scm_to_locale_string (name);
    }

  email = mu_get_user_email (username);
  free (username);
  if (!email)
    mu_scm_error (FUNC_NAME, 0,
		  "Cannot get user email for ~A",
		  scm_list_1 (name));

  ret = scm_from_locale_string (email);
  free (email);
  return ret;
}
#undef FUNC_NAME

void
mu_scm_address_init ()
{
#include <mu_address.x>
}
