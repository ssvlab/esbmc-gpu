/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>

#include <mailutils/errno.h>
#include <auth0.h>

static int
_authenticate_null (mu_authority_t auth MU_ARG_UNUSED)
{
  return 0;
}

int
mu_authority_create_null (mu_authority_t *pauthority, void *owner)
{
  int rc = mu_authority_create(pauthority, NULL, owner);
  if (rc)
    return rc;
  mu_authority_set_authenticate (*pauthority, _authenticate_null,  owner);
  return 0;
}

int
mu_authority_create (mu_authority_t *pauthority, mu_ticket_t ticket, void *owner)
{
  mu_authority_t authority;
  if (pauthority == NULL)
    return MU_ERR_OUT_PTR_NULL;
  authority = calloc (1, sizeof (*authority));
  if (authority == NULL)
    return ENOMEM;
  authority->ticket = ticket;
  authority->owner = owner;
  *pauthority = authority;
  return 0;
}

void
mu_authority_destroy (mu_authority_t *pauthority, void *owner)
{
  if (pauthority && *pauthority)
    {
      mu_authority_t authority = *pauthority;
      if (authority->owner == owner)
	{
	  mu_ticket_destroy (&authority->ticket);
	  mu_list_destroy (&authority->auth_methods);
	  free (authority);
	}
      *pauthority = NULL;
    }
}

void *
mu_authority_get_owner (mu_authority_t authority)
{
  return (authority) ? authority->owner : NULL;
}

int
mu_authority_set_ticket (mu_authority_t authority, mu_ticket_t ticket)
{
  if (authority == NULL)
    return EINVAL;
  if (authority->ticket)
    mu_ticket_destroy (&authority->ticket);
  authority->ticket = ticket;
  return 0;
}

int
mu_authority_get_ticket (mu_authority_t authority, mu_ticket_t *pticket)
{
  if (authority == NULL)
    return EINVAL;
  if (pticket == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (authority->ticket == NULL)
    {
      int status = mu_ticket_create (&(authority->ticket), authority);
      if (status != 0)
	return status;
    }
  *pticket = authority->ticket;
  return 0;
}

struct auth_cb
{
  int status;
  mu_authority_t authority;
};

static int
try_auth (void *item, void *data)
{
  int (*authenticate) (mu_authority_t) = item;
  struct auth_cb *cb = data;
  if (authenticate (cb->authority) == 0)
    {
      cb->status = 0;
      return 1;
    }
  return 0;
}

int
mu_authority_authenticate (mu_authority_t authority)
{
  if (authority && authority->auth_methods)
    {
      struct auth_cb cb;
      cb.status = MU_ERR_AUTH_FAILURE;
      cb.authority = authority;
      mu_list_do (authority->auth_methods, try_auth, &cb);
      return cb.status;
    }
  return EINVAL;
}

int
mu_authority_set_authenticate (mu_authority_t authority,
			    int (*_authenticate) (mu_authority_t),
			    void *owner)
{
  if (authority == NULL)
    return EINVAL;

  if (authority->owner != owner)
    return EACCES;
  if (!authority->auth_methods)
    {
      int rc = mu_list_create (&authority->auth_methods);
      if (rc)
	return rc;
    }
  mu_list_append (authority->auth_methods, _authenticate);
  return 0;
}
