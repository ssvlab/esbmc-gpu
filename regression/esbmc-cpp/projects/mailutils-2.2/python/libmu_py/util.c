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

#define PY_MODULE "util"

static PyObject *
api_util_get_user_email (PyObject *self, PyObject *args)
{
  char *name = NULL;
  char *email = NULL;

  if (!PyArg_ParseTuple (args, "|s", &name))
    return NULL;

  email = mu_get_user_email (name);
  return _ro (PyString_FromString (email ? email : ""));
}

static PyObject *
api_util_set_user_email (PyObject *self, PyObject *args)
{
  int status;
  char *email;

  if (!PyArg_ParseTuple (args, "s", &email))
    return NULL;

  status = mu_set_user_email (email);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_util_get_user_email_domain (PyObject *self, PyObject *args)
{
  int status;
  const char *domain = NULL;

  status = mu_get_user_email_domain (&domain);
  return status_object (status, PyString_FromString (domain ? domain : ""));
}

static PyObject *
api_util_set_user_email_domain (PyObject *self, PyObject *args)
{
  int status;
  char *domain;

  if (!PyArg_ParseTuple (args, "s", &domain))
    return NULL;

  status = mu_set_user_email_domain (domain);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_util_tempname (PyObject *self, PyObject *args)
{
  char *tmpdir = NULL, *tmpname = NULL;

  if (!PyArg_ParseTuple (args, "|z", &tmpdir))
    return NULL;

  tmpname = mu_tempname (tmpdir);
  return _ro (PyString_FromString (tmpname ? tmpname : ""));
}

static PyMethodDef methods[] = {
  { "get_user_email", (PyCFunction) api_util_get_user_email, METH_VARARGS,
    "Get the default user email address." },

  { "set_user_email", (PyCFunction) api_util_set_user_email, METH_VARARGS,
    "Set the default user email address." },

  { "get_user_email_domain", (PyCFunction) api_util_get_user_email_domain,
    METH_VARARGS,
    "Get the default user email address domain." },

  { "set_user_email_domain", (PyCFunction) api_util_set_user_email_domain,
    METH_VARARGS,
    "Set the default user email address domain." },

  { "tempname", (PyCFunction) api_util_tempname,
    METH_VARARGS, "" },

  { NULL, NULL, 0, NULL }
};

void
_mu_py_attach_util (void)
{
  _mu_py_attach_module (PY_MODULE, methods);
}
