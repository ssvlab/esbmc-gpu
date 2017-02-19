/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2004, 2005, 2007, 2010 Free Software
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <errno.h>
#include <stdlib.h>

#include <list0.h>
#include <iterator0.h>
#include <mailutils/errno.h>

int
mu_iterator_create (mu_iterator_t *piterator, void *owner)
{
  mu_iterator_t iterator;
  if (piterator == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (owner == NULL)
    return EINVAL;
  iterator = calloc (1, sizeof (*iterator));
  if (iterator == NULL)
    return ENOMEM;
  iterator->owner = owner;
  *piterator = iterator;
  return 0;
}

int
mu_iterator_set_first (mu_iterator_t itr, int (*first) (void *))
{
  if (!itr)
    return EINVAL;
  itr->first = first;
  return 0;
}

int
mu_iterator_set_next (mu_iterator_t itr, int (*next) (void *))
{
  if (!itr)
    return EINVAL;
  itr->next = next;
  return 0;
}

int
mu_iterator_set_getitem (mu_iterator_t itr,
                         int (*getitem) (void *, void **, const void **))
{
  if (!itr)
    return EINVAL;
  itr->getitem = getitem;
  return 0;
}

int
mu_iterator_set_finished_p (mu_iterator_t itr, int (*finished_p) (void *))
{
  if (!itr)
    return EINVAL;
  itr->finished_p = finished_p;
  return 0;
}

int
mu_iterator_set_curitem_p (mu_iterator_t itr,
			int (*curitem_p) (void *, void *))
{
  if (!itr)
    return EINVAL;
  itr->curitem_p = curitem_p;
  return 0;
}

int
mu_iterator_set_itrctl (mu_iterator_t itr,
			int (*itrctl) (void *,
				       enum mu_itrctl_req,
				       void *))
{
  if (!itr)
    return EINVAL;
  itr->itrctl = itrctl;
  return 0;
}

int
mu_iterator_set_destroy (mu_iterator_t itr, int (*destroy) (mu_iterator_t, void *))
{
  if (!itr)
    return EINVAL;
  itr->destroy = destroy;
  return 0;
}

int
mu_iterator_set_dup (mu_iterator_t itr, int (*dup) (void **ptr, void *data))
{
  if (!itr)
    return EINVAL;
  itr->dup = dup;
  return 0;
}



int
mu_iterator_dup (mu_iterator_t *piterator, mu_iterator_t orig)
{
  mu_iterator_t iterator;
  int status;
  
  if (piterator == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (orig == NULL)
    return EINVAL;

  status = mu_iterator_create (&iterator, orig->owner);
  if (status)
    return status;

  status = orig->dup(&iterator->owner, orig->owner);
  if (status)
    {
      free (iterator);
      return status;
    }
  iterator->is_advanced = orig->is_advanced;   
  iterator->dup = orig->dup;
  iterator->destroy = orig->destroy;
  iterator->first = orig->first;
  iterator->next = orig->next;
  iterator->getitem = orig->getitem;
  iterator->curitem_p = orig->curitem_p;
  iterator->finished_p = orig->finished_p;
  iterator->itrctl = orig->itrctl;
  
  *piterator = iterator;
  return 0;
}

void
mu_iterator_destroy (mu_iterator_t *piterator)
{
  if (!piterator || !*piterator)
    return;

  if ((*piterator)->destroy)
    (*piterator)->destroy (*piterator, (*piterator)->owner);
  
  free (*piterator);
  *piterator = NULL;
}

int
mu_iterator_first (mu_iterator_t iterator)
{
  iterator->is_advanced = 0;
  return iterator->first (iterator->owner);
}

int
mu_iterator_next (mu_iterator_t iterator)
{
  int status = 0;
  if (!iterator->is_advanced)
    status = iterator->next (iterator->owner);
  iterator->is_advanced = 0;
  return status;
}

int
mu_iterator_skip (mu_iterator_t iterator, ssize_t count)
{
  int status;
  if (count < 0)
    return ENOSYS; /* Need prev method */
  while (count--)
    if ((status = mu_iterator_next (iterator)))
      break;
  return status;
}

int
mu_iterator_current (mu_iterator_t iterator, void **pitem)
{
  return iterator->getitem (iterator->owner, pitem, NULL);
}

int
mu_iterator_current_kv (mu_iterator_t iterator, 
                        const void **pkey, void **pitem)
{
  return iterator->getitem (iterator->owner, (void**)pitem, pkey);
}

int
mu_iterator_is_done (mu_iterator_t iterator)
{
  if (iterator == NULL)
    return 1;
  return iterator->finished_p (iterator->owner);
}

int
iterator_get_owner (mu_iterator_t iterator, void **powner)
{
  if (!iterator)
    return EINVAL;
  if (!powner)
    return MU_ERR_OUT_PTR_NULL;
  *powner = iterator->owner;
  return 0;
}

void
mu_iterator_advance (mu_iterator_t iterator, void *e)
{
  for (; iterator; iterator = iterator->next_itr)
    {
      if (iterator->curitem_p (iterator->owner, e))
	{
	  iterator->next (iterator->owner);
	  iterator->is_advanced++;
	}
    }
}

int
mu_iterator_attach (mu_iterator_t *root, mu_iterator_t iterator)
{
  iterator->next_itr = *root;
  *root = iterator;
  return 0;
}

int
mu_iterator_detach (mu_iterator_t *root, mu_iterator_t iterator)
{
  mu_iterator_t itr, prev;
  
  for (itr = *root, prev = NULL; itr; prev = itr, itr = itr->next_itr)
    if (iterator == itr)
      break;

  if (itr)
    {
      if (prev)
	prev->next_itr = itr->next_itr;
      else
	*root = itr->next_itr;
    }
  
  return 0;
}

int
mu_iterator_ctl (mu_iterator_t iterator, enum mu_itrctl_req req, void *arg)
{
  if (!iterator)
    return EINVAL;
  if (!iterator->itrctl)
    return ENOSYS;
  return iterator->itrctl (iterator->owner, req, arg);
}
