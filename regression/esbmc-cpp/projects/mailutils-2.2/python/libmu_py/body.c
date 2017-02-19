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

#define PY_MODULE "body"
#define PY_CSNAME "BodyType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyBodyType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyBody),           /* tp_basicsize */
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

PyBody *
PyBody_NEW ()
{
  return (PyBody *)PyObject_NEW (PyBody, &PyBodyType);
}

static PyObject *
api_body_size (PyObject *self, PyObject *args)
{
  int status;
  size_t size;
  PyBody *py_body;

  if (!PyArg_ParseTuple (args, "O!", &PyBodyType, &py_body))
    return NULL;

  status = mu_body_size (py_body->body, &size);
  return status_object (status, PyInt_FromLong (size));
}

static PyObject *
api_body_lines (PyObject *self, PyObject *args)
{
  int status;
  size_t lines;
  PyBody *py_body;

  if (!PyArg_ParseTuple (args, "O!", &PyBodyType, &py_body))
    return NULL;

  status = mu_body_lines (py_body->body, &lines);
  return status_object (status, PyInt_FromLong (lines));
}

static PyObject *
api_body_get_stream (PyObject *self, PyObject *args)
{
  int status;
  PyBody *py_body;
  PyStream *py_stm = PyStream_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyBodyType, &py_body))
    return NULL;

  Py_INCREF (py_stm);

  status = mu_body_get_stream (py_body->body, &py_stm->stm);
  return status_object (status, (PyObject *)py_stm);
}

static PyMethodDef methods[] = {
  { "size", (PyCFunction) api_body_size, METH_VARARGS,
    "Retrieve 'body' size." },

  { "lines", (PyCFunction) api_body_lines, METH_VARARGS,
    "Retrieve 'body' number of lines." },

  { "get_stream", (PyCFunction) api_body_get_stream, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_body (void)
{
  PyBodyType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyBodyType);
}

void
_mu_py_attach_body (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyBodyType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyBodyType);
    }
}
