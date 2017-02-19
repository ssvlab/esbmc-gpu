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

#ifndef _MUCPP_ITERATOR_H
#define _MUCPP_ITERATOR_H

#include <errno.h>
#include <mailutils/iterator.h>
#include <mailutils/cpp/error.h>
#include <mailutils/cpp/list.h>

namespace mailutils
{

class List;

class Iterator
{
 protected:
  mu_iterator_t mu_iter;
  List* pList;

  friend class List;

 public:
  Iterator (const List&);
  Iterator (const mu_iterator_t);
  ~Iterator ();

  bool operator == (const Iterator&);
  bool operator != (const Iterator&);

  void dup (Iterator*&, const Iterator&);
  void first ();
  void next ();
  Iterator& operator ++ (int);
  void current (void** pitem);
  void* current ();
  bool is_done ();
  List& get_list ();

  inline void* operator * () {
    return this->current ();
  }
};

}

#endif // not _MUCPP_ITERATOR_H

