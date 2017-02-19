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

#define PY_MODULE "mime"
#define PY_CSNAME "MimeType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMimeType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyMime),           /* tp_basicsize */
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

PyMime *
PyMime_NEW ()
{
  return (PyMime *)PyObject_NEW (PyMime, &PyMimeType);
}

static PyObject *
api_mime_create (PyObject *self, PyObject *args)
{
  int status, flags;
  PyMime *py_mime;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!Oi", &PyMimeType, &py_mime,
			 &py_msg, &flags))
    return NULL;

  if (!PyMessage_Check ((PyObject *)py_msg))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_mime_create (&py_mime->mime, py_msg->msg, flags);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mime_destroy (PyObject *self, PyObject *args)
{
  PyMime *py_mime;

  if (!PyArg_ParseTuple (args, "O!", &PyMimeType, &py_mime))
    return NULL;

  mu_mime_destroy (&py_mime->mime);
  return _ro (Py_None);
}

static PyObject *
api_mime_is_multipart (PyObject *self, PyObject *args)
{
  int ismulti;
  PyMime *py_mime;

  if (!PyArg_ParseTuple (args, "O!", &PyMimeType, &py_mime))
    return NULL;

  ismulti = mu_mime_is_multipart (py_mime->mime);
  return _ro (PyBool_FromLong (ismulti));
}

static PyObject *
api_mime_get_num_parts (PyObject *self, PyObject *args)
{
  int status;
  size_t nparts;
  PyMime *py_mime;

  if (!PyArg_ParseTuple (args, "O!", &PyMimeType, &py_mime))
    return NULL;

  status = mu_mime_get_num_parts (py_mime->mime, &nparts);
  return status_object (status, PyInt_FromLong (nparts));
}

static PyObject *
api_mime_get_part (PyObject *self, PyObject *args)
{
  int status;
  size_t npart;
  PyMime *py_mime;
  PyMessage *py_part = PyMessage_NEW ();

  if (!PyArg_ParseTuple (args, "O!i", &PyMimeType, &py_mime, &npart))
    return NULL;

  status = mu_mime_get_part (py_mime->mime, npart, &py_part->msg);

  Py_INCREF (py_part);
  return status_object (status, (PyObject *)py_part);
}

static PyObject *
api_mime_add_part (PyObject *self, PyObject *args)
{
  int status;
  PyMime *py_mime;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!O", &PyMimeType, &py_mime, &py_msg))
    return NULL;

  if (!PyMessage_Check ((PyObject *)py_msg))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_mime_add_part (py_mime->mime, py_msg->msg);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mime_get_message (PyObject *self, PyObject *args)
{
  int status;
  PyMime *py_mime;
  PyMessage *py_msg = PyMessage_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMimeType, &py_mime))
    return NULL;

  status = mu_mime_get_message (py_mime->mime, &py_msg->msg);

  Py_INCREF (py_msg);
  return status_object (status, (PyObject *)py_msg);
}

static PyObject *
api_rfc2047_decode (PyObject *self, PyObject *args)
{
  int status;
  char *tocode, *text;
  char *buf = NULL;

  if (!PyArg_ParseTuple (args, "ss", &tocode, &text))
    return NULL;

  status = mu_rfc2047_decode (tocode, text, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyObject *
api_rfc2047_encode (PyObject *self, PyObject *args)
{
  int status;
  char *charset, *encoding, *text;
  char *buf = NULL;

  if (!PyArg_ParseTuple (args, "sss", &charset, &encoding, &text))
    return NULL;

  status = mu_rfc2047_encode (charset, encoding, text, &buf);
  return status_object (status, PyString_FromString (buf ? buf : ""));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_mime_create, METH_VARARGS,
    "" },

  { "destroy", (PyCFunction) api_mime_destroy, METH_VARARGS,
    "" },

  { "is_multipart", (PyCFunction) api_mime_is_multipart, METH_VARARGS,
    "" },

  { "get_num_parts", (PyCFunction) api_mime_get_num_parts, METH_VARARGS,
    "" },

  { "get_part", (PyCFunction) api_mime_get_part, METH_VARARGS,
    "" },

  { "add_part", (PyCFunction) api_mime_add_part, METH_VARARGS,
    "" },

  { "get_message", (PyCFunction) api_mime_get_message, METH_VARARGS,
    "" },

  { "rfc2047_decode", (PyCFunction) api_rfc2047_decode, METH_VARARGS,
    "" },

  { "rfc2047_encode", (PyCFunction) api_rfc2047_encode, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_mime (void)
{
  PyMimeType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyMimeType);
}

void
_mu_py_attach_mime (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyMimeType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyMimeType);
    }
}
