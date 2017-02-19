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

#define PY_MODULE "url"
#define PY_CSNAME "UrlType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyUrlType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyUrl),            /* tp_basicsize */
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

PyUrl *
PyUrl_NEW ()
{
  return (PyUrl *)PyObject_NEW (PyUrl, &PyUrlType);
}

static PyObject *
api_url_create (PyObject *self, PyObject *args)
{
  int status;
  char *str;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!s", &PyUrlType, &py_url, &str))
    return NULL;

  status = mu_url_create (&py_url->url, str);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_url_destroy (PyObject *self, PyObject *args)
{
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  mu_url_destroy (&py_url->url);
  return _ro (Py_None);
}

static PyObject *
api_url_parse (PyObject *self, PyObject *args)
{
  int status;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_parse (py_url->url);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_url_get_port (PyObject *self, PyObject *args)
{
  int status;
  long port;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_get_port (py_url->url, &port);
  return status_object (status, PyInt_FromLong (port));
}

static PyObject *
api_url_get_scheme (PyObject *self, PyObject *args)
{
  int status;
  const char *buf = NULL;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_scheme (py_url->url, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_url_get_user (PyObject *self, PyObject *args)
{
  int status;
  const char *buf = NULL;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_user (py_url->url, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_url_get_secret (PyObject *self, PyObject *args)
{
  int status;
  PyUrl *py_url;
  PySecret *py_secret = PySecret_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  Py_INCREF (py_secret);

  status = mu_url_get_secret (py_url->url, &py_secret->secret);
  return status_object (status, (PyObject *)py_secret);
}

static PyObject *
api_url_get_auth (PyObject *self, PyObject *args)
{
  int status;
  const char *buf = NULL;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_auth (py_url->url, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_url_get_host (PyObject *self, PyObject *args)
{
  int status;
  const char *buf = NULL;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_host (py_url->url, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_url_get_path (PyObject *self, PyObject *args)
{
  int status;
  const char *buf = NULL;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_path (py_url->url, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_url_get_query (PyObject *self, PyObject *args)
{
  int status, i;
  size_t argc;
  char **argv;
  PyObject *py_list;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  status = mu_url_sget_query (py_url->url, &argc, &argv);

  py_list = PyList_New (0);
  for (i = 0; i < argc; i++)
    PyList_Append (py_list, PyString_FromString (argv[i]));

  return status_object (status, py_list);
}

static PyObject *
api_url_to_string (PyObject *self, PyObject *args)
{
  const char *str;
  PyUrl *py_url;

  if (!PyArg_ParseTuple (args, "O!", &PyUrlType, &py_url))
    return NULL;

  str = mu_url_to_string (py_url->url);
  return _ro (PyString_FromString (str ? str : ""));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_url_create, METH_VARARGS,
    "Create the url data structure, but do not parse it." },

  { "destroy", (PyCFunction) api_url_destroy, METH_VARARGS,
    "Destroy the url and free its resources." },

  { "parse", (PyCFunction) api_url_parse, METH_VARARGS,
    "Parse the url, after calling this the get functions can be called." },

  { "to_string", (PyCFunction) api_url_to_string, METH_VARARGS,
    "" },

  { "get_port", (PyCFunction) api_url_get_port, METH_VARARGS, "" },
  { "get_scheme", (PyCFunction) api_url_get_scheme, METH_VARARGS, "" },
  { "get_user", (PyCFunction) api_url_get_user, METH_VARARGS, "" },
  { "get_secret", (PyCFunction) api_url_get_secret, METH_VARARGS, "" },
  { "get_auth", (PyCFunction) api_url_get_auth, METH_VARARGS, "" },
  { "get_host", (PyCFunction) api_url_get_host, METH_VARARGS, "" },
  { "get_path", (PyCFunction) api_url_get_path, METH_VARARGS, "" },
  { "get_query", (PyCFunction) api_url_get_query, METH_VARARGS, "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_url (void)
{
  PyUrlType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyUrlType);
}

void
_mu_py_attach_url (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyUrlType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyUrlType);
    }
}
