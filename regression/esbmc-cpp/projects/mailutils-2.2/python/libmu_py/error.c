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

#define PY_MODULE "error"

static PyObject *
api_strerror (PyObject *self, PyObject *args)
{
  int status;
  char *str = NULL;

  if (!PyArg_ParseTuple (args, "i", &status))
    return NULL;

  str = (char *)mu_strerror (status);
  return _ro (PyString_FromString (str));
}

static PyMethodDef methods[] = {
  { "strerror", (PyCFunction) api_strerror, METH_VARARGS,
    "Return the error message corresponding to 'err', "
    "which must be an integer value." },

  { NULL, NULL, 0, NULL }
};

void
_mu_py_attach_error (void)
{
  _mu_py_attach_module (PY_MODULE, methods);
}
