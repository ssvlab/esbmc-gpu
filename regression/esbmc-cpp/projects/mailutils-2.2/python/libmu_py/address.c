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

#define PY_MODULE "address"
#define PY_CSNAME "AddressType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyAddressType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyAddress),        /* tp_basicsize */
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

PyAddress *
PyAddress_NEW ()
{
  return (PyAddress *)PyObject_NEW (PyAddress, &PyAddressType);
}

int
PyAddress_Check (PyObject *x)
{
  return x->ob_type == &PyAddressType;
}

static PyObject *
api_address_create (PyObject *self, PyObject *args)
{
  int status;
  char *str;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!s", &PyAddressType, &py_addr, &str))
    return NULL;

  status = mu_address_create (&py_addr->addr, str);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_address_createv (PyObject *self, PyObject *args)
{
  int status;
  PyAddress *py_addr;
  PyObject *py_seq;
  char **sv;
  size_t len;

  if (!PyArg_ParseTuple (args, "O!O", &PyAddressType, &py_addr, &py_seq))
    return NULL;

  if (!PySequence_Check (py_seq))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  len = PySequence_Size (py_seq);
  sv = (char **) malloc ((len + 1) * sizeof (char *));
  if (!sv)
    {
      PyErr_NoMemory ();
      return NULL;
    }
  else
    {
      PyObject *py_item;
      int i;

      for (i = 0; i < len; i++) {
	py_item = PySequence_GetItem (py_seq, i);
	if (py_item && PyString_Check (py_item))
	  sv[i] = strdup (PyString_AsString (py_item));
	Py_DECREF (py_item);
      }
      if (PyErr_Occurred ()) {
	PyErr_Print ();
	return NULL;
      }
    }
  status = mu_address_createv (&py_addr->addr, (const char**)sv, len);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_address_destroy (PyObject *self, PyObject *args)
{
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!", &PyAddressType, &py_addr))
    return NULL;

  mu_address_destroy (&py_addr->addr);
  return _ro (Py_None);
}

static PyObject *
api_address_is_group (PyObject *self, PyObject *args)
{
  int status, n, isgroup;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_is_group (py_addr->addr, n, &isgroup);
  return status_object (status, PyBool_FromLong (isgroup));
}

static PyObject *
api_address_get_count (PyObject *self, PyObject *args)
{
  size_t count = 0;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!", &PyAddressType, &py_addr))
    return NULL;

  mu_address_get_count (py_addr->addr, &count);
  return _ro (PyInt_FromLong (count));
}

static PyObject *
api_address_get_email (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_email (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_get_local_part (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_local_part (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_get_domain (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_domain (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_get_personal (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_personal (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_get_comments (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_comments (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_get_route (PyObject *self, PyObject *args)
{
  int status, n;
  const char *buf = NULL;
  PyAddress *py_addr;

  if (!PyArg_ParseTuple (args, "O!i", &PyAddressType, &py_addr, &n))
    return NULL;

  status = mu_address_sget_route (py_addr->addr, n, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_address_to_string (PyObject *self, PyObject *args)
{
  int status;
  size_t n;
  char buf[256];
  PyAddress *py_addr;

  memset (buf, 0, sizeof (buf));

  if (!PyArg_ParseTuple (args, "O!", &PyAddressType, &py_addr))
    return NULL;

  status = mu_address_to_string (py_addr->addr, buf, sizeof (buf), &n);
  return status_object (status, PyString_FromString (buf));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_address_create, METH_VARARGS,
    "Allocate and initialize 'addr' by parsing the RFC822 "
    "address-list 'string'." },

  { "createv", (PyCFunction) api_address_createv, METH_VARARGS,
    "Allocate and initialize 'addr' by parsing the RFC822 address-list." },

  { "destroy", (PyCFunction) api_address_destroy, METH_VARARGS,
    "Destroy 'addr'." },

  { "is_group", (PyCFunction) api_address_is_group, METH_VARARGS,
    "Return True if address is just the name of a group, False otherwise." },

  { "get_count", (PyCFunction) api_address_get_count, METH_VARARGS,
    "Return a count of the addresses in the address list." },

  { "get_email", (PyCFunction) api_address_get_email, METH_VARARGS,
    "Return the Nth email address component of the address list." },

  { "get_local_part", (PyCFunction) api_address_get_local_part, METH_VARARGS,
    "Return local part of the Nth email address from the address list." },

  { "get_domain", (PyCFunction) api_address_get_domain, METH_VARARGS,
    "Return domain part of the Nth email address from the address list." },

  { "get_personal", (PyCFunction) api_address_get_personal, METH_VARARGS,
    "Return personal part of the Nth email address from the address list." },

  { "get_comments", (PyCFunction) api_address_get_comments, METH_VARARGS,
    "Return comment part of the Nth email address from the list." },

  { "get_route", (PyCFunction) api_address_get_route, METH_VARARGS,
    "Return route part of the Nth email address from the list." },

  { "to_string", (PyCFunction) api_address_to_string, METH_VARARGS,
    "Return the entire address list as a single RFC822 formatted address list." },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_address (void)
{
  PyAddressType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyAddressType);
}

void
_mu_py_attach_address (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyAddressType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyAddressType);
    }
}
