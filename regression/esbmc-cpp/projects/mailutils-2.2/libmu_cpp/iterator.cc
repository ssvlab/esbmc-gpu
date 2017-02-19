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

#include <mailutils/cpp/iterator.h>

using namespace mailutils;

//
// Iterator
//

Iterator :: Iterator (const List& lst)
{
  int status = mu_list_get_iterator (lst.mu_list, &mu_iter);
  if (status)
    throw Exception ("Iterator::Iterator", status);

  this->pList = (List*) &lst;
}

Iterator :: Iterator (const mu_iterator_t iter)
{
  if (iter == 0)
    throw Exception ("Iterator::Iterator", EINVAL);

  this->mu_iter = iter;
  this->pList = 0;
}

Iterator :: ~Iterator ()
{
  mu_iterator_destroy (&mu_iter);
}

bool
Iterator :: operator == (const Iterator& iter)
{
  return mu_iter == iter.mu_iter;
}

bool
Iterator :: operator != (const Iterator& iter)
{
  return mu_iter != iter.mu_iter;
}

void
Iterator :: first ()
{
  mu_iterator_first (mu_iter);
}

void
Iterator :: next ()
{
  mu_iterator_next (mu_iter);
}

Iterator&
Iterator :: operator ++ (int)
{
  mu_iterator_next (mu_iter);
  return *this;
}

void
Iterator :: current (void** pitem)
{
  int status = mu_iterator_current (mu_iter, pitem);
  if (status)
    throw Exception ("Iterator::current", status);
}

void*
Iterator :: current ()
{
  void* pitem;

  int status = mu_iterator_current (mu_iter, &pitem);
  if (status)
    throw Exception ("Iterator::current", status);

  return pitem;
}

bool
Iterator :: is_done ()
{
  return (bool) mu_iterator_is_done (mu_iter);
}

List&
Iterator :: get_list ()
{
  if (!pList)
    throw Exception ("Iterator::get_list", ENOTSUP);
  return *pList;
}

void
Iterator :: dup (Iterator*& piter, const Iterator& orig)
{
  mu_iterator_t iter;

  int status = mu_iterator_dup (&iter, orig.mu_iter);
  if (status)
    throw Exception ("Iterator::dup", status);

  piter->mu_iter = iter;
}

