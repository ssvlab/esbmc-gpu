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

#ifndef _MUCPP_LIST_H
#define _MUCPP_LIST_H

#include <list>
#include <errno.h>
#include <mailutils/list.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/iterator.h>

typedef int mu_list_action_t (void*, void*);
typedef int (*mu_list_comparator_t) (const void*, const void*);

namespace mailutils
{

std::list<void*> mulist_to_stl (mu_list_t mu_list);

class Iterator;

class List
{
 protected:
  mu_list_t mu_list;
  Iterator* iter;

  friend class Iterator;

 public:
  List ();
  List (const mu_list_t);
  ~List ();

  void append (void* item);
  void prepend (void* item);
  void insert (void* item, void* new_item, int insert_before);
  void remove (void* item);
  void replace (void* old_item, void* new_item);

  void  get (size_t index, void** pitem);
  void* get (size_t index);
  void* front ();
  void* back ();

  Iterator begin ();

  void to_array (void** array, size_t count, size_t* pcount);
  void locate (void* item, void** ret_item);

  void apply (mu_list_action_t* action, void* cbdata);
  mu_list_comparator_t set_comparator (mu_list_comparator_t comp);
  mu_list_destroy_item_t set_destroy_item (mu_list_destroy_item_t mu_destroy_item);

  bool is_empty ();
  size_t count ();
  size_t size ();

  std::list<void*> to_stl ();

  inline void* operator [] (size_t index) {
    return this->get (index);
  }
};

}

#endif // not _MUCPP_LIST_H

