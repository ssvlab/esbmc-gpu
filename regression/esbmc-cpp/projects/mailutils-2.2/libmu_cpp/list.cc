/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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
   Boston, MA 02110-1301 USA
*/

#include <mailutils/cpp/list.h>

using namespace mailutils;

std::list<void*>
mailutils :: mulist_to_stl (mu_list_t mu_list)
{
  size_t list_count;
  std::list<void *> list;

  if (!mu_list)
    return list;

  int status = mu_list_count (mu_list, &list_count);
  if (status)
    return list;

  for (int i = 0; i < list_count; i++)
    {
      void *item = NULL;
      status = mu_list_get (mu_list, i, &item);
      if (!status && item)
	list.push_back (item);
    }
  return list;
}

//
// List
//

List :: List ()
{
  int status = mu_list_create (&mu_list);
  if (status)
    throw Exception ("List::List", status);
}

List :: List (const mu_list_t lst)
{
  if (lst == 0)
    throw Exception ("List::List", EINVAL);

  this->mu_list = lst;
}

List :: ~List ()
{
  mu_list_destroy (&mu_list);
}

void
List :: append (void* item)
{
  int status = mu_list_append (mu_list, item);
  if (status)
    throw Exception ("List::append", status);
}

void
List :: prepend (void* item)
{
  int status = mu_list_prepend (mu_list, item);
  if (status)
    throw Exception ("List::prepend", status);
}

void
List :: insert (void* item, void* new_item, int insert_before)
{
  int status = mu_list_insert (mu_list, item, new_item, insert_before);
  if (status)
    throw Exception ("List::insert", status);
}

void
List :: remove (void* item)
{
  int status = mu_list_remove (mu_list, item);
  if (status)
    throw Exception ("List::remove", status);
}

void
List :: replace (void* old_item, void* new_item)
{
  int status = mu_list_replace (mu_list, old_item, new_item);
  if (status)
    throw Exception ("List::replace", status);
}

void
List :: get (size_t index, void** pitem)
{
  int status = mu_list_get (mu_list, index, pitem);
  if (status)
    throw Exception ("List::get", status);
}

void*
List :: get (size_t index)
{
  void* pitem;

  int status = mu_list_get (mu_list, index, &pitem);
  if (status)
    throw Exception ("List::get", status);

  return pitem;
}

inline void*
List :: front ()
{
  return this->get (0);
}

void*
List :: back ()
{
  size_t count = this->count ();
  if (count)
    return this->get (count - 1);
  else
    return NULL;
}

Iterator
List :: begin ()
{
  mu_iterator_t mu_iter;
  int status = mu_list_get_iterator (this->mu_list, &mu_iter);
  if (status)
    throw Exception ("Iterator::begin", status);

  Iterator itr = Iterator (mu_iter);

  this->iter = &itr;
  this->iter->first ();
  return itr;
}

void
List :: to_array (void** array, size_t count, size_t* pcount)
{
  int status = mu_list_to_array (mu_list, array, count, pcount);
  if (status)
    throw Exception ("List::to_array", status);
}

void
List :: locate (void* item, void** ret_item)
{
  int status = mu_list_locate (mu_list, item, ret_item);
  if (status)
    throw Exception ("List::locate", status);
}

bool
List :: is_empty ()
{
  return (bool) mu_list_is_empty (mu_list);
}

size_t
List :: count ()
{
  size_t count = 0;

  int status = mu_list_count (mu_list, &count);
  if (status)
    throw Exception ("List::count", status);

  return count;
}

inline size_t
List :: size ()
{
  return this->count ();
}

void
List :: apply (mu_list_action_t* action, void* cbdata)
{
  int status = mu_list_do (mu_list, action, cbdata);
  if (status)
    throw Exception ("List::apply", status);
}

mu_list_comparator_t
List :: set_comparator (mu_list_comparator_t comp)
{
  return mu_list_set_comparator (mu_list, comp);
}

mu_list_destroy_item_t
List :: set_destroy_item (mu_list_destroy_item_t mu_destroy_item)
{
  return mu_list_set_destroy_item (mu_list, mu_destroy_item);
}

std::list<void*>
List :: to_stl ()
{
  return mulist_to_stl (mu_list);
}

