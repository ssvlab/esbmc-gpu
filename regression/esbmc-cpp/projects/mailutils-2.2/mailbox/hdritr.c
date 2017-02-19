/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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

/* Mail header iterators. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <string.h>

#include <header0.h>
#include <mailutils/errno.h>

struct header_iterator
{
  mu_header_t header;
  size_t index;
};

static int
hdr_first (void *owner)
{
  struct header_iterator *itr = owner;
  itr->index = 1;
  return 0;
}

static int
hdr_next (void *owner)
{
  struct header_iterator *itr = owner;
  itr->index++;
  return 0;
}

static int
hdr_getitem (void *owner, void **pret, const void **pkey)
{
  struct header_iterator *itr = owner;
  int rc;
  size_t count;

  rc = mu_header_get_field_count (itr->header, &count);
  if (rc)
    return rc;
  if (itr->index > count)
    return MU_ERR_NOENT;
  
  rc = mu_header_sget_field_name (itr->header, itr->index,
				  (const char**) pkey);
  if (rc == 0)
    {
      if (pkey)
	rc = mu_header_sget_field_value (itr->header, itr->index,
					 (const char**) pret);
    }
  return rc;
}

static int
hdr_finished_p (void *owner)
{
  struct header_iterator *itr = owner;
  size_t count;

  if (mu_header_get_field_count (itr->header, &count))
    return 1;
  return itr->index > count;
}

static int
hdr_destroy (mu_iterator_t iterator, void *data)
{
  struct header_iterator *itr = data;
  mu_iterator_detach (&itr->header->itr, iterator);
  free (data);
  return 0;
}

static int
hdr_curitem_p (void *owner, void *item)
{
  void *ptr;

  if (hdr_getitem (owner, &ptr, NULL))
    return 0;
  return ptr == item;
}

static int
hdr_data_dup (void **ptr, void *owner)
{
  struct header_iterator *itr = owner;

  *ptr = malloc (sizeof (struct header_iterator));
  if (*ptr == NULL)
    return ENOMEM;
  memcpy (*ptr, owner, sizeof (struct header_iterator));
  mu_iterator_attach (&itr->header->itr, *ptr);
  return 0;
}

int
mu_header_get_iterator (mu_header_t hdr, mu_iterator_t *piterator)
{
  mu_iterator_t iterator;
  int status;
  struct header_iterator *itr;

  if (!hdr)
    return EINVAL;

  itr = calloc (1, sizeof *itr);
  if (!itr)
    return ENOMEM;
  itr->header = hdr;
  itr->index = 1;

  status = mu_iterator_create (&iterator, itr);
  if (status)
    {
      free (itr);
      return status;
    }

  mu_iterator_set_first (iterator, hdr_first);
  mu_iterator_set_next (iterator, hdr_next);
  mu_iterator_set_getitem (iterator, hdr_getitem);
  mu_iterator_set_finished_p (iterator, hdr_finished_p);
  mu_iterator_set_curitem_p (iterator, hdr_curitem_p);
  mu_iterator_set_destroy (iterator, hdr_destroy);
  mu_iterator_set_dup (iterator, hdr_data_dup);

  mu_iterator_attach (&hdr->itr, iterator);

  *piterator = iterator;
  return 0;
}








