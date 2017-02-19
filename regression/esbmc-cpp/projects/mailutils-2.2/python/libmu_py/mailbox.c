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

#define PY_MODULE "mailbox"
#define PY_CSNAME "MailboxType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMailboxType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyMailbox),        /* tp_basicsize */
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

PyMailbox *
PyMailbox_NEW ()
{
  return (PyMailbox *)PyObject_NEW (PyMailbox, &PyMailboxType);
}

int
PyMailbox_Check (PyObject *x)
{
  return x->ob_type == &PyMailboxType;
}

static PyObject *
api_mailbox_create (PyObject *self, PyObject *args)
{
  int status;
  char *name;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!s", &PyMailboxType, &py_mbox, &name))
    return NULL;

  status = mu_mailbox_create (&py_mbox->mbox, name);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_create_default (PyObject *self, PyObject *args)
{
  int status;
  char *name;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!z", &PyMailboxType, &py_mbox, &name))
    return NULL;

  status = mu_mailbox_create_default (&py_mbox->mbox, name);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_destroy (PyObject *self, PyObject *args)
{
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  mu_mailbox_destroy (&py_mbox->mbox);
  return _ro (Py_None);
}

static PyObject *
api_mailbox_open (PyObject *self, PyObject *args)
{
  int status;
  int flag;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!i", &PyMailboxType, &py_mbox, &flag))
    return NULL;

  status = mu_mailbox_open (py_mbox->mbox, flag);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_close (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_close (py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_flush (PyObject *self, PyObject *args)
{
  int status, expunge = 0;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!i", &PyMailboxType, &py_mbox, &expunge))
    return NULL;

  status = mu_mailbox_flush (py_mbox->mbox, expunge);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_messages_count (PyObject *self, PyObject *args)
{
  int status;
  size_t total = 0;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_messages_count (py_mbox->mbox, &total);
  return status_object (status, PyInt_FromLong (total));
}

static PyObject *
api_mailbox_messages_recent (PyObject *self, PyObject *args)
{
  int status;
  size_t recent = 0;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_messages_recent (py_mbox->mbox, &recent);
  return status_object (status, PyInt_FromLong (recent));
}

static PyObject *
api_mailbox_message_unseen (PyObject *self, PyObject *args)
{
  int status;
  size_t unseen = 0;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_message_unseen (py_mbox->mbox, &unseen);
  return status_object (status, PyInt_FromLong (unseen));
}

static PyObject *
api_mailbox_get_message (PyObject *self, PyObject *args)
{
  int status;
  size_t msgno;
  PyMailbox *py_mbox;
  PyMessage *py_msg = PyMessage_NEW ();

  if (!PyArg_ParseTuple (args, "O!i", &PyMailboxType, &py_mbox, &msgno))
    return NULL;

  status = mu_mailbox_get_message (py_mbox->mbox, msgno, &py_msg->msg);

  Py_INCREF (py_msg);
  return status_object (status, (PyObject *)py_msg);
}

static PyObject *
api_mailbox_append_message (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!O", &PyMailboxType, &py_mbox, &py_msg))
    return NULL;

  if (!PyMessage_Check ((PyObject *)py_msg))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_mailbox_append_message (py_mbox->mbox, py_msg->msg);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_expunge (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_expunge (py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_sync (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_sync (py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static int
uidls_extractor (void *data, PyObject **dst)
{
  struct mu_uidl *uidl = (struct mu_uidl *)data;

  *dst = PyTuple_New (2);
  PyTuple_SetItem (*dst, 0, PyInt_FromLong (uidl->msgno));
  PyTuple_SetItem (*dst, 1, PyString_FromString (uidl->uidl));
  return 0;
}

static PyObject *
api_mailbox_get_uidls (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;
  PyObject *py_list;
  mu_list_t c_list = NULL;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_get_uidls (py_mbox->mbox, &c_list);

  if (c_list)
    py_list = mu_py_mulist_to_pylist (c_list, uidls_extractor);
  else
    py_list = PyTuple_New (0);

  return status_object (status, py_list);
}

static PyObject *
api_mailbox_lock (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_lock (py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_unlock (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_unlock (py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailbox_get_size (PyObject *self, PyObject *args)
{
  int status;
  mu_off_t size = 0;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_get_size (py_mbox->mbox, &size);
  return status_object (status, PyInt_FromLong (size));
}

static PyObject *
api_mailbox_get_debug (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;
  PyDebug *py_dbg = PyDebug_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  status = mu_mailbox_get_debug (py_mbox->mbox, &py_dbg->dbg);

  Py_INCREF (py_dbg);
  return status_object (status, (PyObject *)py_dbg);
}

static PyObject *
api_mailbox_get_folder (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;
  PyFolder *py_folder = PyFolder_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  py_folder->folder = NULL;
  status = mu_mailbox_get_folder (py_mbox->mbox, &py_folder->folder);

  Py_INCREF (py_folder);
  return status_object (status, (PyObject *)py_folder);
}

static PyObject *
api_mailbox_get_url (PyObject *self, PyObject *args)
{
  int status;
  PyMailbox *py_mbox;
  PyUrl *py_url = PyUrl_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyMailboxType, &py_mbox))
    return NULL;

  Py_INCREF (py_url);

  status = mu_mailbox_get_url (py_mbox->mbox, &py_url->url);
  return status_object (status, (PyObject *)py_url);
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_mailbox_create, METH_VARARGS,
    "Allocate and initialize 'mbox'. The concrete mailbox type "
    "instantiate is based on the scheme of the url 'name'." },

  { "create_default", (PyCFunction) api_mailbox_create_default, METH_VARARGS,
    "Create a mailbox with mu_mailbox_create() based on the "
    "environment variable MAIL or the string formed by "
    "__PATH_MAILDIR_/USER or LOGNAME if USER is null." },

  { "destroy", (PyCFunction) api_mailbox_destroy, METH_VARARGS,
    "Destroy and release resources held by 'mbox'." },

  { "open", (PyCFunction) api_mailbox_open, METH_VARARGS,
    "A connection is open, if no stream was provided, a stream is "
    "created based on the 'mbox' type. The 'flag' can be OR'ed." },

  { "close", (PyCFunction) api_mailbox_close, METH_VARARGS,
    "The stream attached to 'mbox' is closed." },

  { "flush", (PyCFunction) api_mailbox_flush, METH_VARARGS,
    "" },

  { "messages_count", (PyCFunction) api_mailbox_messages_count, METH_VARARGS,
    "Give the number of messages in 'mbox'." },

  { "messages_recent", (PyCFunction) api_mailbox_messages_recent, METH_VARARGS,
    "Give the number of recent messages in 'mbox'." },

  { "message_unseen", (PyCFunction) api_mailbox_message_unseen, METH_VARARGS,
    "Give the number of first unseen message in MBOX." },

  { "get_message", (PyCFunction) api_mailbox_get_message, METH_VARARGS,
    "Retrieve message number 'msgno', 'message' is allocated and initialized." },

  { "append_message", (PyCFunction) api_mailbox_append_message, METH_VARARGS,
    "Append 'message' to the mailbox 'mbox'." },

  { "expunge", (PyCFunction) api_mailbox_expunge, METH_VARARGS,
    "Expunge deleted messages from the mailbox 'mbox'." },

  { "sync", (PyCFunction) api_mailbox_sync, METH_VARARGS,
    "" },

  { "get_uidls", (PyCFunction) api_mailbox_get_uidls, METH_VARARGS,
    "" },

  { "lock", (PyCFunction) api_mailbox_lock, METH_VARARGS,
    "" },

  { "unlock", (PyCFunction) api_mailbox_unlock, METH_VARARGS,
    "" },

  { "get_size", (PyCFunction) api_mailbox_get_size, METH_VARARGS,
    "" },

  { "get_debug", (PyCFunction) api_mailbox_get_debug, METH_VARARGS,
    "" },

  { "get_folder", (PyCFunction) api_mailbox_get_folder, METH_VARARGS,
    "" },

  { "get_url", (PyCFunction) api_mailbox_get_url, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_mailbox (void)
{
  PyMailboxType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyMailboxType);
}

void
_mu_py_attach_mailbox (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyMailboxType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyMailboxType);
    }
}
