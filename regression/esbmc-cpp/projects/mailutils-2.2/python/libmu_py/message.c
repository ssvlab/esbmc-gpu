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

#define PY_MODULE "message"
#define PY_CSNAME "MessageType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMessageType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyMessage),        /* tp_basicsize */
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

PyMessage *
PyMessage_NEW ()
{
  return (PyMessage *)PyObject_NEW (PyMessage, &PyMessageType);
}

int
PyMessage_Check (PyObject *x)
{
  return x->ob_type == &PyMessageType;
}

static PyObject *
api_message_create (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_create (&py_msg->msg, NULL);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_message_destroy (PyObject *self, PyObject *args)
{
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  mu_message_destroy (&py_msg->msg, NULL);
  return _ro (Py_None);
}

static PyObject *
api_message_is_multipart (PyObject *self, PyObject *args)
{
  int status, ismulti;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_is_multipart (py_msg->msg, &ismulti);
  return status_object (status, PyBool_FromLong (ismulti));
}

static PyObject *
api_message_size (PyObject *self, PyObject *args)
{
  int status;
  size_t size;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_size (py_msg->msg, &size);
  return status_object (status, PyInt_FromLong (size));
}

static PyObject *
api_message_lines (PyObject *self, PyObject *args)
{
  int status;
  size_t lines;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_lines (py_msg->msg, &lines);
  return status_object (status, PyInt_FromLong (lines));
}

static PyObject *
api_message_get_envelope (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyEnvelope *py_env = PyEnvelope_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_envelope (py_msg->msg, &py_env->env);

  Py_INCREF (py_env);
  return status_object (status, (PyObject *)py_env);
}

static PyObject *
api_message_get_header (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyHeader *py_hdr = PyHeader_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_header (py_msg->msg, &py_hdr->hdr);

  Py_INCREF (py_hdr);
  return status_object (status, (PyObject *)py_hdr);
}

static PyObject *
api_message_get_body (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyBody *py_body = PyBody_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_body (py_msg->msg, &py_body->body);

  Py_INCREF (py_body);
  return status_object (status, (PyObject *)py_body);
}

static PyObject *
api_message_get_attribute (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyAttribute *py_attr = PyAttribute_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_attribute (py_msg->msg, &py_attr->attr);

  Py_INCREF (py_attr);
  return status_object (status, (PyObject *)py_attr);
}

static PyObject *
api_message_get_num_parts (PyObject *self, PyObject *args)
{
  int status;
  size_t parts;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_num_parts (py_msg->msg, &parts);
  return status_object (status, PyInt_FromLong (parts));
}

static PyObject *
api_message_get_part (PyObject *self, PyObject *args)
{
  int status;
  size_t npart;
  PyMessage *py_msg;
  PyMessage *py_part = PyMessage_NEW ();

  if (!PyArg_ParseTuple (args, "O!i", &PyMessageType, &py_msg, &npart))
    return NULL;

  status = mu_message_get_part (py_msg->msg, npart, &py_part->msg);

  Py_INCREF (py_part);
  return status_object (status, (PyObject *)py_part);
}

static PyObject *
api_message_get_uid (PyObject *self, PyObject *args)
{
  int status;
  size_t uid;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_uid (py_msg->msg, &uid);
  return status_object (status, PyInt_FromLong (uid));
}

static PyObject *
api_message_get_uidl (PyObject *self, PyObject *args)
{
  int status;
  char buf[512];
  size_t writen;
  PyMessage *py_msg;

  memset (buf, 0, sizeof (buf));

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  status = mu_message_get_uidl (py_msg->msg, buf, sizeof (buf), &writen);
  return status_object (status, PyString_FromString (buf));
}

static PyObject *
api_message_get_attachment_name (PyObject *self, PyObject *args)
{
  int status;
  char *name = NULL;
  char *charset = NULL;
  char *lang = NULL;
  PyObject *py_ret;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!|z", &PyMessageType, &py_msg, &charset))
    return NULL;

  status = mu_message_aget_decoded_attachment_name (py_msg->msg, charset,
						    &name, &lang);

  py_ret = PyTuple_New (3);
  PyTuple_SetItem (py_ret, 0, PyInt_FromLong (status));
  PyTuple_SetItem (py_ret, 1, PyString_FromString (name ? name : ""));
  PyTuple_SetItem (py_ret, 2, lang ? PyString_FromString (lang) : Py_None);
  return _ro (py_ret);
}

static PyObject *
api_message_save_attachment (PyObject *self, PyObject *args)
{
  int status;
  char *filename = NULL;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!|s", &PyMessageType, &py_msg,
			 &filename))
    return NULL;

  if (!strlen (filename))
    filename = NULL;

  status = mu_message_save_attachment (py_msg->msg, filename, NULL);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_message_unencapsulate (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyMessage *py_unen = PyMessage_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMessageType, &py_msg))
    return NULL;

  Py_INCREF (py_unen);

  status = mu_message_unencapsulate (py_msg->msg, &py_unen->msg, NULL);
  return status_object (status, (PyObject *)py_unen);
}

static PyObject *
api_message_set_stream (PyObject *self, PyObject *args)
{
  int status;
  PyMessage *py_msg;
  PyStream *py_stm;

  if (!PyArg_ParseTuple (args, "O!O", &PyMessageType, &py_msg, &py_stm))
    return NULL;

  if (!PyStream_Check ((PyObject *)py_stm))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_message_set_stream (py_msg->msg, py_stm->stm, NULL);
  py_stm->stm = NULL;

  return _ro (PyInt_FromLong (status));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_message_create, METH_VARARGS,
    "Create message." },

  { "destroy", (PyCFunction) api_message_destroy, METH_VARARGS,
    "The resources allocate for 'msg' are freed." },

  { "is_multipart", (PyCFunction) api_message_is_multipart, METH_VARARGS,
    "" },

  { "size", (PyCFunction) api_message_size, METH_VARARGS,
    "Retrieve 'msg' size." },

  { "lines", (PyCFunction) api_message_lines, METH_VARARGS,
    "Retrieve 'msg' number of lines." },

  { "get_envelope", (PyCFunction) api_message_get_envelope, METH_VARARGS,
    "Retrieve 'msg' envelope." },

  { "get_header", (PyCFunction) api_message_get_header, METH_VARARGS,
    "Retrieve 'msg' header." },

  { "get_body", (PyCFunction) api_message_get_body, METH_VARARGS,
    "Retrieve 'msg' body." },

  { "get_attribute", (PyCFunction) api_message_get_attribute, METH_VARARGS,
    "Retrieve 'msg' attribute." },

  { "get_num_parts", (PyCFunction) api_message_get_num_parts, METH_VARARGS,
    "" },

  { "get_part", (PyCFunction) api_message_get_part, METH_VARARGS,
    "" },

  { "get_uid", (PyCFunction) api_message_get_uid, METH_VARARGS,
    "" },

  { "get_uidl", (PyCFunction) api_message_get_uidl, METH_VARARGS,
    "" },

  { "get_attachment_name", (PyCFunction) api_message_get_attachment_name,
    METH_VARARGS, "" },

  { "save_attachment", (PyCFunction) api_message_save_attachment,
    METH_VARARGS, "" },

  { "unencapsulate", (PyCFunction) api_message_unencapsulate,
    METH_VARARGS, "" },

  { "set_stream", (PyCFunction) api_message_set_stream, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_message (void)
{
  PyMessageType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyMessageType);
}

void
_mu_py_attach_message (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyMessageType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyMessageType);
    }
}
