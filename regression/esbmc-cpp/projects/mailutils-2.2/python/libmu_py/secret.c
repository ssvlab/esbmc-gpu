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

#define PY_MODULE "secret"
#define PY_CSNAME "SecretType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PySecretType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PySecret),         /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)_py_dealloc,   /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr; __getattr__ */
  0,                         /* tp_setattr; __setattr__ */
  0,                         /* tp_compare; __cmp__ */
  _repr,                     /* tp_repr; __repr__ */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash; __hash__ */
  0,                         /* tp_call; __call__ */
  _repr,                     /* tp_str; __str__ */
  0,                         /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,        /* tp_flags */
  "",                        /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  0,                         /* tp_methods */
  0,                         /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  0,                         /* tp_init */
  0,                         /* tp_alloc */
  0,                         /* tp_new */
};

PySecret *
PySecret_NEW ()
{
  return (PySecret *)PyObject_NEW (PySecret, &PySecretType);
}

int
PySecret_Check (PyObject *x)
{
  return x->ob_type == &PySecretType;
}

static PyObject *
api_secret_create (PyObject *self, PyObject *args)
{
  int status;
  char *str;
  size_t len;
  PySecret *py_secret;

  if (!PyArg_ParseTuple (args, "O!si", &PySecretType, &py_secret,
			 &str, &len))
    return NULL;

  status = mu_secret_create (&py_secret->secret, str, len);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_secret_destroy (PyObject *self, PyObject *args)
{
  PySecret *py_secret;

  if (!PyArg_ParseTuple (args, "O!", &PySecretType, &py_secret))
    return NULL;

  mu_secret_destroy (&py_secret->secret);
  return _ro (Py_None);
}

static PyObject *
api_secret_password (PyObject *self, PyObject *args)
{
  const char *pass;
  PySecret *py_secret;

  if (!PyArg_ParseTuple (args, "O!", &PySecretType, &py_secret))
    return NULL;

  pass = mu_secret_password (py_secret->secret);
  return _ro (PyString_FromString (pass ? pass : ""));
}

static PyObject *
api_secret_password_unref (PyObject *self, PyObject *args)
{
  PySecret *py_secret;

  if (!PyArg_ParseTuple (args, "O!", &PySecretType, &py_secret))
    return NULL;

  mu_secret_password_unref (py_secret->secret);
  return _ro (Py_None);
}

static PyObject *
api_clear_passwd (PyObject *self, PyObject *args)
{
  char *p;

  if (!PyArg_ParseTuple (args, "s", &p))
    return NULL;

  while (*p)
    *p++ = 0;
  return _ro (Py_None);
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_secret_create, METH_VARARGS,
    "Create the secret data structure." },

  { "destroy", (PyCFunction) api_secret_destroy, METH_VARARGS,
    "Destroy the secret and free its resources." },

  { "password", (PyCFunction) api_secret_password, METH_VARARGS,
    "" },

  { "password_unref", (PyCFunction) api_secret_password_unref, METH_VARARGS,
    "" },

  { "clear_passwd", (PyCFunction) api_clear_passwd, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_secret (void)
{
  PySecretType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PySecretType);
}

void
_mu_py_attach_secret (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PySecretType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PySecretType);
    }
}
