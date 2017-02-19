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

#define PY_MODULE "mailer"
#define PY_CSNAME "MailerType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMailerType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyMailer),         /* tp_basicsize */
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

PyMailer *
PyMailer_NEW ()
{
  return (PyMailer *)PyObject_NEW (PyMailer, &PyMailerType);
}

static PyObject *
api_mailer_create (PyObject *self, PyObject *args)
{
  int status;
  char *url;
  PyMailer *py_mlr;

  if (!PyArg_ParseTuple (args, "O!s", &PyMailerType, &py_mlr, &url))
    return NULL;

  status = mu_mailer_create (&py_mlr->mlr, url);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailer_destroy (PyObject *self, PyObject *args)
{
  PyMailer *py_mlr;

  if (!PyArg_ParseTuple (args, "O!", &PyMailerType, &py_mlr))
    return NULL;

  mu_mailer_destroy (&py_mlr->mlr);
  return _ro (Py_None);
}

static PyObject *
api_mailer_open (PyObject *self, PyObject *args)
{
  int status, flags;
  PyMailer *py_mlr;

  if (!PyArg_ParseTuple (args, "O!i", &PyMailerType, &py_mlr, &flags))
    return NULL;

  status = mu_mailer_open (py_mlr->mlr, flags);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailer_close (PyObject *self, PyObject *args)
{
  int status;
  PyMailer *py_mlr;

  if (!PyArg_ParseTuple (args, "O!", &PyMailerType, &py_mlr))
    return NULL;

  status = mu_mailer_close (py_mlr->mlr);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailer_send_message (PyObject *self, PyObject *args)
{
  int status;
  PyMailer *py_mlr;
  PyMessage *py_msg;
  PyAddress *py_from, *py_to;
  mu_address_t c_from = NULL, c_to = NULL;

  if (!PyArg_ParseTuple (args, "O!OOO", &PyMailerType, &py_mlr,
			 &py_msg, &py_from, &py_to))
    return NULL;

  if (!PyMessage_Check ((PyObject *)py_msg))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }
  if (!PyAddress_Check ((PyObject *)py_from) &&
      (PyObject *)py_from != Py_None)
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }
  if (!PyAddress_Check ((PyObject *)py_to) &&
      (PyObject *)py_to != Py_None)
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }
  if ((PyObject *)py_from != Py_None)
    c_from = py_from->addr;
  if ((PyObject *)py_to != Py_None)
    c_to = py_to->addr;

  status = mu_mailer_send_message (py_mlr->mlr, py_msg->msg,
				   c_from, c_to);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailer_get_debug (PyObject *self, PyObject *args)
{
  int status;
  PyMailer *py_mlr;
  PyDebug *py_dbg = PyDebug_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMailerType, &py_mlr))
    return NULL;

  status = mu_mailer_get_debug (py_mlr->mlr, &py_dbg->dbg);

  Py_INCREF (py_dbg);
  return status_object (status, (PyObject *)py_dbg);
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_mailer_create, METH_VARARGS,
    "Create mailer." },

  { "destroy", (PyCFunction) api_mailer_destroy, METH_VARARGS,
    "The resources allocate for 'msg' are freed." },

  { "open", (PyCFunction) api_mailer_open, METH_VARARGS,
    "" },

  { "close", (PyCFunction) api_mailer_close, METH_VARARGS,
    "" },

  { "send_message", (PyCFunction) api_mailer_send_message, METH_VARARGS,
    "" },

  { "get_debug", (PyCFunction) api_mailer_get_debug, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_mailer (void)
{
  PyMailerType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyMailerType);
}

void
_mu_py_attach_mailer (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyMailerType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyMailerType);
    }
}
