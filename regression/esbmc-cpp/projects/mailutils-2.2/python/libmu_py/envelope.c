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

#define PY_MODULE "envelope"
#define PY_CSNAME "EnvelopeType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyEnvelopeType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyEnvelope),       /* tp_basicsize */
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

PyEnvelope *
PyEnvelope_NEW ()
{
  return (PyEnvelope *)PyObject_NEW (PyEnvelope, &PyEnvelopeType);
}

static PyObject *
api_envelope_create (PyObject *self, PyObject *args)
{
  int status;
  PyEnvelope *py_env;

  if (!PyArg_ParseTuple (args, "O!", &PyEnvelopeType, &py_env))
    return NULL;

  status = mu_envelope_create (&py_env->env, NULL);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_envelope_destroy (PyObject *self, PyObject *args)
{
  PyEnvelope *py_env;

  if (!PyArg_ParseTuple (args, "O!", &PyEnvelopeType, &py_env))
    return NULL;

  mu_envelope_destroy (&py_env->env, NULL);
  return _ro (Py_None);
}

static PyObject *
api_envelope_get_sender (PyObject *self, PyObject *args)
{
  int status;
  const char *sender = NULL;
  PyEnvelope *py_env;

  if (!PyArg_ParseTuple (args, "O!", &PyEnvelopeType, &py_env))
    return NULL;

  status = mu_envelope_sget_sender (py_env->env, &sender);
  return status_object (status, PyString_FromString (sender ? sender : ""));
}

static PyObject *
api_envelope_get_date (PyObject *self, PyObject *args)
{
  int status;
  const char *date = NULL;
  PyEnvelope *py_env;

  if (!PyArg_ParseTuple (args, "O!", &PyEnvelopeType, &py_env))
    return NULL;

  status = mu_envelope_sget_date (py_env->env, &date);
  return status_object (status, PyString_FromString (date ? date : ""));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_envelope_create, METH_VARARGS,
    "" },

  { "destroy", (PyCFunction) api_envelope_destroy, METH_VARARGS,
    "" },

  { "get_sender", (PyCFunction) api_envelope_get_sender, METH_VARARGS,
    "Get the address that this message was reportedly received from." },

  { "get_date", (PyCFunction) api_envelope_get_date, METH_VARARGS,
    "Get the date that the message was delivered to the mailbox." },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_envelope ()
{
  PyEnvelopeType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyEnvelopeType);
}

void
_mu_py_attach_envelope (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyEnvelopeType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyEnvelopeType);
    }
}
