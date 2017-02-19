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

#define PY_MODULE "nls"

static PyObject *
api_nls_init (PyObject *self)
{
  mu_init_nls ();
  return _ro (Py_None);
}

static PyObject *
api_nls_set_locale (PyObject *self, PyObject *args)
{
  char *lset;
  const char *locale;

  if (!PyArg_ParseTuple (args, "s", &locale))
    return NULL;

  lset = mu_set_locale (locale);
  return _ro (PyString_FromString (lset ? lset : ""));
}

static PyObject *
api_nls_restore_locale (PyObject *self)
{
  mu_restore_locale ();
  return _ro (Py_None);
}

static PyMethodDef methods[] = {
  { "init", (PyCFunction) api_nls_init, METH_NOARGS,
    "Initialize Native Language Support." },

  { "set_locale", (PyCFunction) api_nls_set_locale, METH_VARARGS,
    "Set locale via LC_ALL." },

  { "restore_locale", (PyCFunction) api_nls_restore_locale, METH_NOARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

void
_mu_py_attach_nls (void)
{
  _mu_py_attach_module (PY_MODULE, methods);
}
