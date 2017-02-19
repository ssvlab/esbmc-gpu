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

void
_mu_list_insert_sublist (mu_list_t list,
			 struct list_data *current,
			 struct list_data *head,
			 struct list_data *tail,
			 size_t count,
			 int insert_before)
{
  if (insert_before)
    {
      head->prev = current->prev;
      tail->next = current;
      if (current->prev != &list->head)
	current->prev->next = head;
      else
	list->head.next = head;

      current->prev = tail;
    }
  else
    {
      tail->next = current->next;
      head->prev = current;
      if (current->next != &list->head)
	current->next->prev = tail;
      else
	list->head.prev = tail;

      current->next = head;
    }
  list->count += count;
}

void
_mu_list_clear (mu_list_t list)
{
  list->head.next = list->head.prev = &list->head;
  list->count = 0;
}

int
mu_list_insert_list (mu_list_t list, void *item, mu_list_t new_list,
		     int insert_before)
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
	  _mu_list_insert_sublist (list, current,
				   new_list->head.next, new_list->head.prev,
				   new_list->count,
				   insert_before);
	  _mu_list_clear (new_list);
	  status = 0;
	  break;
	}
    }
  mu_monitor_unlock (list->monitor);
  return status;
}

void
mu_list_append_list (mu_list_t list, mu_list_t new_list)
{
  if (list->count == 0)
    {
      list->head = new_list->head;
      list->head.next->prev = list->head.prev->next = &list->head;
      list->count = new_list->count;
    }
  else
    _mu_list_insert_sublist (list, list->head.prev,
			     new_list->head.next, new_list->head.prev,
			     new_list->count,
			     0);
  _mu_list_clear (new_list);
}

void
mu_list_prepend_list (mu_list_t list, mu_list_t new_list)
{
  if (list->count == 0)
    {
      list->head = new_list->head;
      list->head.next->prev = list->head.prev->next = &list->head;
      list->count = new_list->count;
    }
  else
    _mu_list_insert_sublist (list, list->head.next,
			     new_list->head.next, new_list->head.prev,
			     new_list->count,
			     1);
  _mu_list_clear (new_list);
}



