/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2008, 2010 Free
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
#include <config.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <property0.h>
#include <mailutils/errno.h>
#include <mailutils/assoc.h>
#include <stdlib.h>

static void
prop_free_value (void *data)
{
  struct property_item *item = data;
  free (item->value);
}

int
mu_property_create (mu_property_t *pp, void *owner)
{
  int rc;
  mu_property_t prop;
  if (pp == NULL)
    return MU_ERR_OUT_PTR_NULL;
  prop = calloc (1, sizeof *prop);
  if (prop == NULL)
    return ENOMEM;
  rc = mu_assoc_create (&prop->assoc, sizeof (struct property_item), 0);
  if (rc)
    {
      free (prop);
      return rc;
    }
  mu_assoc_set_free (prop->assoc, prop_free_value);
  mu_monitor_create (&prop->lock, 0, prop);
  prop->owner = owner;
  *pp = prop;
  return 0;
}

void
mu_property_destroy (mu_property_t *pp, void *owner)
{
  if (pp && *pp)
    {
      mu_property_t prop = *pp;
      if (prop->owner == owner)
	{
	  /* Destroy the list and is properties.  */
	  mu_monitor_wrlock (prop->lock);
	  mu_assoc_destroy (&prop->assoc);
	  mu_monitor_unlock (prop->lock);
	  mu_monitor_destroy (&prop->lock, prop);
	  free (prop);
	}
      *pp = NULL;
    }
}

void *
mu_property_get_owner (mu_property_t prop)
{
  return (prop == NULL) ? NULL : prop->owner;
}

int
mu_property_set_value (mu_property_t prop, const char *key, const char *value,
		       int overwrite)
{
  struct property_item *item;
  int rc;

  if (!prop)
    return EINVAL;
  rc = mu_assoc_ref_install (prop->assoc, key, (void **)&item);
  if (rc == MU_ERR_NOENT)
    {
      item->value = strdup (value);
    }
  else if (overwrite)
    {
      free (item->value);
      item->value = strdup (value);
    }
  else
    return 0;
  
  if (!item->value)
    {
      mu_assoc_remove (prop->assoc, key);
      return ENOMEM;
    }

  return 0;
}
  
int
mu_property_sget_value (mu_property_t prop, const char *key,
			const char **buffer)
{
  struct property_item *item;

  if (!prop)
    return EINVAL;
  item = mu_assoc_ref (prop->assoc, key);
  if (item == NULL)
    return MU_ERR_NOENT;
  if (buffer)
    *buffer = item->value;
  return 0;
}

int
mu_property_aget_value (mu_property_t prop, const char *key,
			char **buffer)
{
  const char *value;
  int rc = mu_property_sget_value (prop, key, &value);
  if (rc == 0)
    {
      if ((*buffer = strdup (value)) == NULL)
	return ENOMEM;
    }
  return rc;
}

int
mu_property_get_value (mu_property_t prop, const char *key, char *buffer,
		       size_t buflen, size_t *n)
{
  size_t len = 0;
  const char *value;
  int rc = mu_property_sget_value (prop, key, &value);
  if (rc == 0)
    {
      len = strlen (value) + 1;
      if (buffer && buflen)
	{
	  if (buflen < len)
	    len = buflen;
	  len--;
	  memcpy (buffer, value, len);
	  buffer[len] = 0;
	}
    }
  if (n)
    *n = len;
  return rc;
}

int
mu_property_is_set (mu_property_t prop, const char *key)
{
  struct property_item *item = mu_assoc_ref (prop->assoc, key);
  return (item == NULL) ? 0 : 1;
}

int
mu_property_set (mu_property_t prop, const char *key)
{
  return mu_property_set_value (prop, key, "", 1);
}

int
mu_property_unset (mu_property_t prop, const char *key)
{
  if (!prop)
    return EINVAL;
  return mu_assoc_remove (prop->assoc, key);
}
  
