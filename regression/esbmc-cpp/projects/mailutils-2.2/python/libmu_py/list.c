/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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

#include "libmu_py.h"

PyObject *
mu_py_mulist_to_pylist (mu_list_t list, mulist_extractor_fp extractor)
{
  int status, i;
  size_t list_count;
  PyObject *py_list;

  if (!list)
    return NULL;

  status = mu_list_count (list, &list_count);
  if (status)
    return NULL;

  py_list = PyList_New (list_count);
  if (!py_list)
    return NULL;

  for (i = 0; i < list_count; i++)
    {
      void *item;
      PyObject *py_item = NULL;

      status = mu_list_get (list, i, &item);
      if (!status)
	{
	  status = (*extractor)(item, &py_item);
	  if (!status && py_item)
	    PyList_SetItem (py_list, i, py_item);
	}
    }
  return py_list;
}
