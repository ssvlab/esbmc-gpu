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
# include <config.h>
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <list0.h>
#include <iterator0.h>
#include <mailutils/errno.h>

#define DESTROY_ITEM(list, elt)			\
  do						\
    {						\
       if ((list)->destroy_item)		\
	 (list)->destroy_item ((elt)->item);	\
    }						\
  while (0)

int
mu_list_create (mu_list_t *plist)
{
  mu_list_t list;
  int status;

  if (plist == NULL)
    return MU_ERR_OUT_PTR_NULL;
  list = calloc (sizeof (*list), 1);
  if (list == NULL)
    return ENOMEM;
  status = mu_monitor_create (&list->monitor, 0,  list);
  if (status != 0)
    {
      free (list);
      return status;
    }
  list->head.next = &list->head;
  list->head.prev = &list->head;
  *plist = list;
  return 0;
}

void
mu_list_destroy (mu_list_t *plist)
{
  if (plist && *plist)
    {
      mu_list_t list = *plist;
      struct list_data *current;
      struct list_data *previous;

      mu_monitor_wrlock (list->monitor);
      for (current = list->head.next; current != &list->head;)
	{
	  previous = current;
	  current = current->next;
	  DESTROY_ITEM (list, previous);
	  free (previous);
	}
      mu_monitor_unlock (list->monitor);
      mu_monitor_destroy (&list->monitor, list);
      free (list);
      *plist = NULL;
    }
}

int
mu_list_append (mu_list_t list, void *item)
{
  struct list_data *ldata;
  struct list_data *last;

  if (list == NULL)
    return EINVAL;
  last = list->head.prev;
  ldata = calloc (sizeof (*ldata), 1);
  if (ldata == NULL)
    return ENOMEM;
  ldata->item = item;
  mu_monitor_wrlock (list->monitor);
  ldata->next = &list->head;
  ldata->prev = list->head.prev;
  last->next = ldata;
  list->head.prev = ldata;
  list->count++;
  mu_monitor_unlock (list->monitor);
  return 0;
}

int
mu_list_prepend (mu_list_t list, void *item)
{
  struct list_data *ldata;
  struct list_data *first;

  if (list == NULL)
    return EINVAL;
  first = list->head.next;
  ldata = calloc (sizeof (*ldata), 1);
  if (ldata == NULL)
    return ENOMEM;
  ldata->item = item;
  mu_monitor_wrlock (list->monitor);
  ldata->prev = &list->head;
  ldata->next = list->head.next;
  first->prev = ldata;
  list->head.next = ldata;
  list->count++;
  mu_monitor_unlock (list->monitor);
  return 0;
}

int
mu_list_is_empty (mu_list_t list)
{
  size_t n = 0;

  mu_list_count (list, &n);
  return (n == 0);
}

int
mu_list_count (mu_list_t list, size_t *pcount)
{
  if (list == NULL)
    return EINVAL;
  if (pcount == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *pcount = list->count;
  return 0;
}

mu_list_comparator_t
mu_list_set_comparator (mu_list_t list, mu_list_comparator_t comp)
{
  mu_list_comparator_t old_comp;

  if (list == NULL)
    return NULL;
  old_comp = list->comp;
  list->comp = comp;
  return old_comp;
}

int
mu_list_get_comparator (mu_list_t list, mu_list_comparator_t *comp)
{
  if (!list)
    return EINVAL;
  *comp = list->comp;
  return 0;
}

int
_mu_list_ptr_comparator (const void *item, const void *value)
{
  return item != value;
}

int
mu_list_locate (mu_list_t list, void *item, void **ret_item)
{
  struct list_data *current, *previous;
  mu_list_comparator_t comp;
  int status = MU_ERR_NOENT;

  if (list == NULL)
    return EINVAL;
  comp = list->comp ? list->comp : _mu_list_ptr_comparator;
  mu_monitor_wrlock (list->monitor);
  for (previous = &list->head, current = list->head.next;
       current != &list->head; previous = current, current = current->next)
    {
      if (comp (current->item, item) == 0)
	{
	  if (ret_item)
	    *ret_item = current->item;
	  status = 0;
	  break;
	}
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

static int
_insert_item (mu_list_t list, struct list_data *current, void *new_item,
	      int insert_before)
{
  int status;
  struct list_data *ldata = calloc (sizeof (*ldata), 1);
  if (ldata == NULL)
    status = ENOMEM;
  else
    {
      ldata->item = new_item;
      _mu_list_insert_sublist (list, current,
			       ldata, ldata,
			       1,
			       insert_before);
      status = 0;
    }
  return status;
}

int
mu_list_insert (mu_list_t list, void *item, void *new_item, int insert_before)
{
  struct list_data *current;
  mu_list_comparator_t comp;
  int status = MU_ERR_NOENT;

  if (list == NULL)
    return EINVAL;
  comp = list->comp ? list->comp : _mu_list_ptr_comparator;

  mu_monitor_wrlock (list->monitor);
  for (current = list->head.next;
       current != &list->head;
       current = current->next)
    {
      if (comp (current->item, item) == 0)
	{
	  status = _insert_item (list, current, new_item, insert_before);
	  break;
	}
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

int
mu_list_remove (mu_list_t list, void *item)
{
  struct list_data *current;
  mu_list_comparator_t comp;
  int status = MU_ERR_NOENT;

  if (list == NULL)
    return EINVAL;
  comp = list->comp ? list->comp : _mu_list_ptr_comparator;
  mu_monitor_wrlock (list->monitor);
  for (current = list->head.next;
       current != &list->head; current = current->next)
    {
      if (comp (current->item, item) == 0)
	{
	  struct list_data *previous = current->prev;
	  
	  mu_iterator_advance (list->itr, current);
	  previous->next = current->next;
	  current->next->prev = previous;
	  DESTROY_ITEM (list, current);
	  free (current);
	  list->count--;
	  status = 0;
	  break;
	}
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

int
mu_list_remove_nd (mu_list_t list, void *item)
{
  mu_list_destroy_item_t dptr = mu_list_set_destroy_item (list, NULL);
  int rc = mu_list_remove (list, item);
  mu_list_set_destroy_item (list, dptr);
  return rc;
}

int
mu_list_replace (mu_list_t list, void *old_item, void *new_item)
{
  struct list_data *current, *previous;
  mu_list_comparator_t comp;
  int status = MU_ERR_NOENT;

  if (list == NULL)
    return EINVAL;
  comp = list->comp ? list->comp : _mu_list_ptr_comparator;
  mu_monitor_wrlock (list->monitor);
  for (previous = &list->head, current = list->head.next;
       current != &list->head; previous = current, current = current->next)
    {
      if (comp (current->item, old_item) == 0)
	{
	  DESTROY_ITEM (list, current);
	  current->item = new_item;
	  status = 0;
	  break;
	}
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

int
mu_list_replace_nd (mu_list_t list, void *item, void *new_item)
{
  mu_list_destroy_item_t dptr = mu_list_set_destroy_item (list, NULL);
  int rc = mu_list_replace (list, item, new_item);
  mu_list_set_destroy_item (list, dptr);
  return rc;
}

int
mu_list_get (mu_list_t list, size_t indx, void **pitem)
{
  struct list_data *current;
  size_t count;
  int status = MU_ERR_NOENT;

  if (list == NULL)
    return EINVAL;
  if (pitem == NULL)
    return MU_ERR_OUT_PTR_NULL;
  mu_monitor_rdlock (list->monitor);
  for (current = list->head.next, count = 0; current != &list->head;
       current = current->next, count++)
    {
      if (count == indx)
        {
          *pitem = current->item;
	  status = 0;
	  break;
        }
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

int
mu_list_do (mu_list_t list, mu_list_action_t *action, void *cbdata)
{
  mu_iterator_t itr;
  int status = 0;

  if (list == NULL || action == NULL)
    return EINVAL;
  status = mu_list_get_iterator (list, &itr);
  if (status)
    return status;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      void *item;
      mu_iterator_current (itr, &item);
      if ((status = action (item, cbdata)))
	break;
    }
  mu_iterator_destroy (&itr);
  return status;
}

mu_list_destroy_item_t
mu_list_set_destroy_item (mu_list_t list, void (*destroy_item)(void *item))
{
  mu_list_destroy_item_t ret = list->destroy_item;
  list->destroy_item = destroy_item;
  return ret;
}

int
mu_list_to_array (mu_list_t list, void **array, size_t count, size_t *pcount)
{
  size_t total = 0;

  if (!list)
    return EINVAL;

  total = (count < list->count) ? count : list->count;

  if (array)
    {
      size_t i;
      struct list_data *current;

      for (i = 0, current = list->head.next;
	   i < total && current != &list->head; current = current->next)
	array[i++] = current->item;
    }
  if (pcount)
    *pcount = total;
  return 0;
}

/* Computes an intersection of two lists and returns it in PDEST.
   The resulting list contains elements from A that are
   also encountered in B (as per comparison function of
   the latter).

   If DUP_ITEM is not NULL, it is used to create copies of
   items to be stored in PDEST.  In this case, the destroy_item
   function of B is also attached to PDEST.  Otherwise, if
   DUP_ITEM is NULL, pointers to elements are stored and
   no destroy_item function is assigned. */
int
mu_list_intersect_dup (mu_list_t *pdest, mu_list_t a, mu_list_t b,
		       int (*dup_item) (void **, void *, void *),
		       void *dup_closure)
{
  mu_list_t dest;
  int rc;
  mu_iterator_t itr;
  
  rc = mu_list_create (&dest);
  if (rc)
    return rc;

  mu_list_set_comparator (dest, b->comp);
  if (dup_item)
    mu_list_set_destroy_item (dest, b->destroy_item);
  
  rc = mu_list_get_iterator (a, &itr);
  if (rc)
    {
      mu_list_destroy (&dest);
      return rc;
    }

  rc = 0;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      void *data;
      mu_iterator_current (itr, &data);
      if (mu_list_locate (b, data, NULL) == 0)
	{
	  void *new_data;
	  if (dup_item && data)
	    {
	      rc = dup_item (&new_data, data, dup_closure);
	      if (rc)
		break;
	    }
	  else
	    new_data = data;
	
	  mu_list_append (dest, new_data); /* FIXME: Check return, and? */
	}
    }
  mu_iterator_destroy (&itr);
  *pdest = dest;
  return rc;
}

int
mu_list_intersect (mu_list_t *pdest, mu_list_t a, mu_list_t b)
{
  return mu_list_intersect_dup (pdest, a, b, NULL, NULL);
}


/* Iterator interface */

struct list_iterator
{
  mu_list_t list;
  struct list_data *cur;
  int backwards; /* true if iterating backwards */
};

static int
first (void *owner)
{
  struct list_iterator *itr = owner;
  if (itr->backwards)
    itr->cur = itr->list->head.prev;
  else
    itr->cur = itr->list->head.next;
  return 0;
}

static int
next (void *owner)
{
  struct list_iterator *itr = owner;
  if (itr->backwards)
    itr->cur = itr->cur->prev;
  else
    itr->cur = itr->cur->next;
  return 0;
}

static int
getitem (void *owner, void **pret, const void **pkey)
{
  struct list_iterator *itr = owner;
  *pret = itr->cur->item;
  if (pkey)
    *pkey = NULL;
  return 0;
}

static int
finished_p (void *owner)
{
  struct list_iterator *itr = owner;
  return itr->cur == &itr->list->head;
}

static int
destroy (mu_iterator_t iterator, void *data)
{
  struct list_iterator *itr = data;
  mu_iterator_detach (&itr->list->itr, iterator);
  free (data);
  return 0;
}

static int
curitem_p (void *owner, void *item)
{
  struct list_iterator *itr = owner;
  return itr->cur == item;
}

static int
list_data_dup (void **ptr, void *owner)
{
  *ptr = malloc (sizeof (struct list_iterator));
  if (*ptr == NULL)
    return ENOMEM;
  memcpy (*ptr, owner, sizeof (struct list_iterator));
  return 0;
}

static int
list_itrctl (void *owner, enum mu_itrctl_req req, void *arg)
{
  struct list_iterator *itr = owner;
  mu_list_t list = itr->list;
  struct list_data *ptr;
  
  if (itr->cur == NULL)
    return MU_ERR_NOENT;
  switch (req)
    {
    case mu_itrctl_tell:
      /* Return current position in the object */
      {
	size_t count;

	for (count = 0, ptr = list->head.next; ptr != &list->head;
	     ptr = ptr->next, count++)
	  {
	    if (ptr == itr->cur)
	      {
		*(size_t*)arg = count;
		return 0;
	      }
	  }
	return MU_ERR_NOENT;
      }
	
    case mu_itrctl_delete:
    case mu_itrctl_delete_nd:
      /* Delete current element */
      {
	struct list_data *prev;
	
	ptr = itr->cur;
	prev = ptr->prev;
	
	mu_iterator_advance (list->itr, ptr);
	prev->next = ptr->next;
	ptr->next->prev = prev;
	if (req == mu_itrctl_delete)
	  DESTROY_ITEM (list, ptr);
	free (ptr);
	list->count--;
      }
      break;
      
    case mu_itrctl_replace:
    case mu_itrctl_replace_nd:
      /* Replace current element */
      if (!arg)
	return EINVAL;
      if (req == mu_itrctl_replace)
	  DESTROY_ITEM (list, ptr);
      ptr = itr->cur;
      ptr->item = arg;
      break;
      
    case mu_itrctl_insert:
      /* Insert new element in the current position */
      if (!arg)
	return EINVAL;
      return _insert_item (list, itr->cur, arg, 0);

    case mu_itrctl_insert_list:
      /* Insert a list of elements */
      if (!arg)
	return EINVAL;
      else
	{
	  mu_list_t new_list = arg;
	  _mu_list_insert_sublist (list, itr->cur,
				   new_list->head.next, new_list->head.prev,
				   new_list->count,
				   0);
	  _mu_list_clear (new_list);
	}
      break;

    case mu_itrctl_qry_direction:
      if (!arg)
	return EINVAL;
      else
	*(int*)arg = itr->backwards;
      break;

    case mu_itrctl_set_direction:
      if (!arg)
	return EINVAL;
      else
	itr->backwards = !!*(int*)arg;
      break;
      
    default:
      return ENOSYS;
    }
  return 0;
}

int
mu_list_get_iterator (mu_list_t list, mu_iterator_t *piterator)
{
  mu_iterator_t iterator;
  int status;
  struct list_iterator *itr;

  if (!list)
    return EINVAL;

  itr = calloc (1, sizeof *itr);
  if (!itr)
    return ENOMEM;
  itr->list = list;
  itr->cur = NULL;

  status = mu_iterator_create (&iterator, itr);
  if (status)
    {
      free (itr);
      return status;
    }

  mu_iterator_set_first (iterator, first);
  mu_iterator_set_next (iterator, next);
  mu_iterator_set_getitem (iterator, getitem);
  mu_iterator_set_finished_p (iterator, finished_p);
  mu_iterator_set_curitem_p (iterator, curitem_p);
  mu_iterator_set_destroy (iterator, destroy);
  mu_iterator_set_dup (iterator, list_data_dup);
  mu_iterator_set_itrctl (iterator, list_itrctl);
  
  mu_iterator_attach (&list->itr, iterator);

  *piterator = iterator;
  return 0;
}
