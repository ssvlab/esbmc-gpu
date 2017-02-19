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

#define PY_MODULE  "mailcap"
#define PY_CSNAME1 "MailcapType"
#define PY_CSNAME2 "MailcapEntryType"

static PyObject *
_repr1 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME1 " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMailcapType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME1,  /* tp_name */
  sizeof (PyMailcap),        /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)_py_dealloc,   /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr; __getattr__ */
  0,                         /* tp_setattr; __setattr__ */
  0,                         /* tp_compare; __cmp__ */
  _repr1,                    /* tp_repr; __repr__ */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash; __hash__ */
  0,                         /* tp_call; __call__ */
  _repr1,                    /* tp_str; __str__ */
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

PyMailcap *
PyMailcap_NEW ()
{
  return (PyMailcap *)PyObject_NEW (PyMailcap, &PyMailcapType);
}

static PyObject *
_repr2 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME2 " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyMailcapEntryType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME2,  /* tp_name */
  sizeof (PyMailcapEntry),   /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)_py_dealloc,   /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr; __getattr__ */
  0,                         /* tp_setattr; __setattr__ */
  0,                         /* tp_compare; __cmp__ */
  _repr2,                    /* tp_repr; __repr__ */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash; __hash__ */
  0,                         /* tp_call; __call__ */
  _repr2,                    /* tp_str; __str__ */
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

PyMailcapEntry *
PyMailcapEntry_NEW ()
{
  return (PyMailcapEntry *)PyObject_NEW (PyMailcapEntry,
					 &PyMailcapEntryType);
}


static PyObject *
api_mailcap_create (PyObject *self, PyObject *args)
{
  int status;
  PyMailcap *py_mc;
  PyStream *py_stm;

  if (!PyArg_ParseTuple (args, "O!O", &PyMailcapType, &py_mc, &py_stm))
    return NULL;

  if (!PyStream_Check ((PyObject *)py_stm))
    {
      PyErr_SetString (PyExc_TypeError, py_stm->ob_type->tp_name);
      return NULL;
    }

  status = mu_mailcap_create (&py_mc->mc, py_stm->stm);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_mailcap_destroy (PyObject *self, PyObject *args)
{
  PyMailcap *py_mc;

  if (!PyArg_ParseTuple (args, "O!", &PyMailcapType, &py_mc))
    return NULL;

  mu_mailcap_destroy (&py_mc->mc);
  return _ro (Py_None);
}

static PyObject *
api_mailcap_entries_count (PyObject *self, PyObject *args)
{
  int status;
  size_t count = 0;
  PyMailcap *py_mc;

  if (!PyArg_ParseTuple (args, "O!", &PyMailcapType, &py_mc))
    return NULL;

  status = mu_mailcap_entries_count (py_mc->mc, &count);
  return status_object (status, PyInt_FromLong (count));
}

static PyObject *
api_mailcap_get_entry (PyObject *self, PyObject *args)
{
  int status, i;
  PyMailcap *py_mc;
  PyMailcapEntry *py_entry = PyMailcapEntry_NEW ();

  if (!PyArg_ParseTuple (args, "O!i", &PyMailcapType, &py_mc, &i))
    return NULL;

  status = mu_mailcap_get_entry (py_mc->mc, i, &py_entry->entry);

  Py_INCREF (py_entry);
  return status_object (status, (PyObject *)py_entry);
}

static PyObject *
api_mailcap_entry_fields_count (PyObject *self, PyObject *args)
{
  int status;
  size_t count;
  PyMailcapEntry *py_entry;

  if (!PyArg_ParseTuple (args, "O!", &PyMailcapEntryType, &py_entry))
    return NULL;

  status = mu_mailcap_entry_fields_count (py_entry->entry, &count);
  return status_object (status, PyInt_FromLong (count));
}

static PyObject *
api_mailcap_entry_get_field (PyObject *self, PyObject *args)
{
  int status, i;
  char buf[256];
  PyMailcapEntry *py_entry;

  if (!PyArg_ParseTuple (args, "O!i", &PyMailcapEntryType, &py_entry,
			 &i))
    return NULL;

  status = mu_mailcap_entry_get_field (py_entry->entry, i, buf,
				       sizeof (buf), NULL);
  return status_object (status, PyString_FromString (buf));
}

static PyObject *
api_mailcap_entry_get_typefield (PyObject *self, PyObject *args)
{
  int status;
  char buf[256];
  PyMailcapEntry *py_entry;

  if (!PyArg_ParseTuple (args, "O!", &PyMailcapEntryType, &py_entry))
    return NULL;

  status = mu_mailcap_entry_get_typefield (py_entry->entry, buf,
					   sizeof (buf), NULL);
  return status_object (status, PyString_FromString (buf));
}

static PyObject *
api_mailcap_entry_get_viewcommand (PyObject *self, PyObject *args)
{
  int status;
  char buf[256];
  PyMailcapEntry *py_entry;

  if (!PyArg_ParseTuple (args, "O!", &PyMailcapEntryType, &py_entry))
    return NULL;

  status = mu_mailcap_entry_get_viewcommand (py_entry->entry, buf,
					     sizeof (buf), NULL);
  return status_object (status, PyString_FromString (buf));
}

static PyMethodDef methods[] = {
  { "create", (PyCFunction) api_mailcap_create, METH_VARARGS,
    "Allocate, parse the buffer from the 'stream' and initializes 'mailcap'." },

  { "destroy", (PyCFunction) api_mailcap_destroy, METH_VARARGS,
    "Release any resources from the mailcap object." },

  { "entries_count", (PyCFunction) api_mailcap_entries_count, METH_VARARGS,
    "Return the number of entries found in the mailcap." },

  { "get_entry", (PyCFunction) api_mailcap_get_entry, METH_VARARGS,
    "Return in 'entry' the mailcap entry of 'no'." },

  { "entry_fields_count", (PyCFunction) api_mailcap_entry_fields_count,
    METH_VARARGS,
    "Return the number of fields found in the entry." },

  { "entry_get_field", (PyCFunction) api_mailcap_entry_get_field,
    METH_VARARGS,
    "" },

  { "entry_get_typefield", (PyCFunction) api_mailcap_entry_get_typefield,
    METH_VARARGS,
    "" },

  { "entry_get_viewcommand",
    (PyCFunction) api_mailcap_entry_get_viewcommand, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_mailcap (void)
{
  PyMailcapType.tp_new = PyType_GenericNew;
  PyMailcapEntryType.tp_new = PyType_GenericNew;
  if (PyType_Ready (&PyMailcapType) < 0)
    return -1;
  if (PyType_Ready (&PyMailcapEntryType) < 0)
    return -1;
  return 0;
}

void
_mu_py_attach_mailcap (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyMailcapType);
      Py_INCREF (&PyMailcapEntryType);
      PyModule_AddObject (m, PY_CSNAME1, (PyObject *)&PyMailcapType);
      PyModule_AddObject (m, PY_CSNAME2, (PyObject *)&PyMailcapEntryType);
    }
}
