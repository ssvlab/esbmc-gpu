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

#define PY_MODULE "sieve"
#define PY_CSNAME "SieveMachineType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PySieveMachineType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PySieveMachine),   /* tp_basicsize */
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

PySieveMachine *
PySieveMachine_NEW ()
{
  return (PySieveMachine *)PyObject_NEW (PySieveMachine, &PySieveMachineType);
}

int
PySieveMachine_Check (PyObject *x)
{
  return x->ob_type == &PySieveMachineType;
}

struct _mu_py_sieve_logger {
  PyObject *py_debug_printer;
  PyObject *py_error_printer;
  PyObject *py_parse_error_printer;
  PyObject *py_action_printer;
};

static PyObject *
api_sieve_machine_init (PyObject *self, PyObject *args)
{
  int status;
  PySieveMachine *py_mach;
  struct _mu_py_sieve_logger *s;

  if (!PyArg_ParseTuple (args, "O!", &PySieveMachineType, &py_mach))
    return NULL;

  if (!(s = calloc (1, sizeof (*s))))
    return NULL;

  status = mu_sieve_machine_init (&py_mach->mach, s);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_sieve_machine_destroy (PyObject *self, PyObject *args)
{
  PySieveMachine *py_mach;

  if (!PyArg_ParseTuple (args, "O!", &PySieveMachineType, &py_mach))
    return NULL;

  if (py_mach->mach) {
    struct _mu_py_sieve_logger *s = mu_sieve_get_data (py_mach->mach);
    if (s)
      free (s);
    mu_sieve_machine_destroy (&py_mach->mach);
  }
  return _ro (Py_None);
}

static PyObject *
api_sieve_compile (PyObject *self, PyObject *args)
{
  int status;
  char *name;
  PySieveMachine *py_mach;

  if (!PyArg_ParseTuple (args, "O!s", &PySieveMachineType, &py_mach, &name))
    return NULL;

  status = mu_sieve_compile (py_mach->mach, name);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_sieve_disass (PyObject *self, PyObject *args)
{
  int status;
  PySieveMachine *py_mach;

  if (!PyArg_ParseTuple (args, "O!", &PySieveMachineType, &py_mach))
    return NULL;

  status = mu_sieve_disass (py_mach->mach);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_sieve_mailbox (PyObject *self, PyObject *args)
{
  int status;
  PySieveMachine *py_mach;
  PyMailbox *py_mbox;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_mbox))
    return NULL;

  if (!PyMailbox_Check ((PyObject *)py_mbox))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_sieve_mailbox (py_mach->mach, py_mbox->mbox);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_sieve_message (PyObject *self, PyObject *args)
{
  int status;
  PySieveMachine *py_mach;
  PyMessage *py_msg;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_msg))
    return NULL;

  if (!PyMessage_Check ((PyObject *)py_msg))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_sieve_message (py_mach->mach, py_msg->msg);
  return _ro (PyInt_FromLong (status));
}

static int
_sieve_error_printer (void *data, const char *fmt, va_list ap)
{
  char *buf = NULL;
  size_t buflen = 0;
  PyObject *py_args = PyTuple_New (1);

  if (py_args)
    {
      struct _mu_py_sieve_logger *s = data;
      PyObject *py_fnc = s->py_error_printer;

      if (mu_vasnprintf (&buf, &buflen, fmt, ap))
	return 0;
      PyTuple_SetItem (py_args, 0, PyString_FromString (buf ? buf : ""));
      if (buf)
	free (buf);

      if (py_fnc && PyCallable_Check (py_fnc))
	PyObject_CallObject (py_fnc, py_args);

      Py_DECREF (py_args);
    }
  return 0;
}

static int
_sieve_parse_error_printer (void *data, const char *filename, int lineno,
			    const char *fmt, va_list ap)
{
  PyObject *py_args;
  PyObject *py_dict;

  py_dict = PyDict_New ();
  if (py_dict)
    {
      char *buf = NULL;
      size_t buflen = 0;

      PyDict_SetItemString (py_dict, "filename",
			    PyString_FromString (filename));
      PyDict_SetItemString (py_dict, "lineno", PyInt_FromLong (lineno));

      if (mu_vasnprintf (&buf, &buflen, fmt, ap))
	return 0;
      PyDict_SetItemString (py_dict, "text",
			    PyString_FromString (buf ? buf : ""));
      if (buf)
	free (buf);

      py_args = PyTuple_New (1);
      if (py_args)
	{
	  struct _mu_py_sieve_logger *s = data;
	  PyObject *py_fnc = s->py_parse_error_printer;

	  Py_INCREF (py_dict);
	  PyTuple_SetItem (py_args, 0, py_dict);

	  if (py_fnc && PyCallable_Check (py_fnc))
	    PyObject_CallObject (py_fnc, py_args);

	  Py_DECREF (py_dict);
	  Py_DECREF (py_args);
	}
    }
  return 0;
}

static int
_sieve_debug_printer (void *data, const char *fmt, va_list ap)
{
  char *buf = NULL;
  size_t buflen = 0;
  PyObject *py_args = PyTuple_New (1);

  if (py_args)
    {
      struct _mu_py_sieve_logger *s = data;
      PyObject *py_fnc = s->py_debug_printer;

      if (mu_vasnprintf (&buf, &buflen, fmt, ap))
	return 0;
      PyTuple_SetItem (py_args, 0, PyString_FromString (buf ? buf : ""));
      if (buf)
	free (buf);

      if (py_fnc && PyCallable_Check (py_fnc))
	PyObject_CallObject (py_fnc, py_args);

      Py_DECREF (py_args);
    }
  return 0;
}

static void
_sieve_action_printer (void *data, const mu_sieve_locus_t *locus,
		       size_t msgno, mu_message_t msg,
		       const char *action, const char *fmt, va_list ap)
{
  PyObject *py_args;
  PyObject *py_dict;

  py_dict = PyDict_New ();
  if (py_dict)
    {
      char *buf = NULL;
      size_t buflen = 0;
      PyMessage *py_msg = PyMessage_NEW ();

      py_msg->msg = msg;
      Py_INCREF (py_msg);

      PyDict_SetItemString (py_dict, "source_file",
			    PyString_FromString (locus->source_file));
      PyDict_SetItemString (py_dict, "source_line",
			    PyInt_FromLong (locus->source_line));
      PyDict_SetItemString (py_dict, "msgno",
			    PyInt_FromLong (msgno));
      PyDict_SetItemString (py_dict, "msg", (PyObject *)py_msg);
      PyDict_SetItemString (py_dict, "action",
			    PyString_FromString (action ? action : ""));

      if (mu_vasnprintf (&buf, &buflen, fmt, ap))
	return;
      PyDict_SetItemString (py_dict, "text",
			    PyString_FromString (buf ? buf : ""));
      if (buf)
	free (buf);

      py_args = PyTuple_New (1);
      if (py_args)
	{
	  struct _mu_py_sieve_logger *s = data;
	  PyObject *py_fnc = s->py_action_printer;

	  Py_INCREF (py_dict);
	  PyTuple_SetItem (py_args, 0, py_dict);

	  if (py_fnc && PyCallable_Check (py_fnc))
	    PyObject_CallObject (py_fnc, py_args);

	  Py_DECREF (py_dict);
	  Py_DECREF (py_args);
	}
    }
}

static PyObject *
api_sieve_set_error (PyObject *self, PyObject *args)
{
  PySieveMachine *py_mach;
  PyObject *py_fnc;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_fnc))
    return NULL;

  if (py_fnc && PyCallable_Check (py_fnc)) {
    mu_sieve_machine_t mach = py_mach->mach;
    struct _mu_py_sieve_logger *s = mu_sieve_get_data (mach);
    s->py_error_printer = py_fnc;
    Py_INCREF (py_fnc);
    mu_sieve_set_error (py_mach->mach, _sieve_error_printer);
  }
  else {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
  }
  return _ro (Py_None);
}

static PyObject *
api_sieve_set_parse_error (PyObject *self, PyObject *args)
{
  PySieveMachine *py_mach;
  PyObject *py_fnc;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_fnc))
    return NULL;

  if (py_fnc && PyCallable_Check (py_fnc)) {
    mu_sieve_machine_t mach = py_mach->mach;
    struct _mu_py_sieve_logger *s = mu_sieve_get_data (mach);
    s->py_parse_error_printer = py_fnc;
    Py_INCREF (py_fnc);
    mu_sieve_set_parse_error (py_mach->mach, _sieve_parse_error_printer);
  }
  else {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
  }
  return _ro (Py_None);
}

static PyObject *
api_sieve_set_debug (PyObject *self, PyObject *args)
{
  PySieveMachine *py_mach;
  PyObject *py_fnc;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_fnc))
    return NULL;

  if (py_fnc && PyCallable_Check (py_fnc)) {
    mu_sieve_machine_t mach = py_mach->mach;
    struct _mu_py_sieve_logger *s = mu_sieve_get_data (mach);
    s->py_debug_printer = py_fnc;
    Py_INCREF (py_fnc);
    mu_sieve_set_debug (py_mach->mach, _sieve_debug_printer);
  }
  else {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
  }
  return _ro (Py_None);
}

static PyObject *
api_sieve_set_logger (PyObject *self, PyObject *args)
{
  PySieveMachine *py_mach;
  PyObject *py_fnc;

  if (!PyArg_ParseTuple (args, "O!O", &PySieveMachineType, &py_mach, &py_fnc))
    return NULL;

  if (py_fnc && PyCallable_Check (py_fnc)) {
    mu_sieve_machine_t mach = py_mach->mach;
    struct _mu_py_sieve_logger *s = mu_sieve_get_data (mach);
    s->py_action_printer = py_fnc;
    Py_INCREF (py_fnc);
    mu_sieve_set_logger (py_mach->mach, _sieve_action_printer);
  }
  else {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
  }
  return _ro (Py_None);
}

static PyMethodDef methods[] = {
  { "machine_init", (PyCFunction) api_sieve_machine_init, METH_VARARGS,
    "Create and initialize new Sieve 'machine'." },

  { "machine_destroy", (PyCFunction) api_sieve_machine_destroy, METH_VARARGS,
    "Destroy Sieve 'machine'." },

  { "compile", (PyCFunction) api_sieve_compile, METH_VARARGS,
    "Compile the sieve script from the file 'name'." },

  { "disass", (PyCFunction) api_sieve_disass, METH_VARARGS,
    "Dump the disassembled code of the sieve machine 'mach'." },

  { "mailbox", (PyCFunction) api_sieve_mailbox, METH_VARARGS,
    "Execute the code from the given instance of sieve machine "
    "'mach' over each message in the mailbox 'mbox'." },

  { "message", (PyCFunction) api_sieve_message, METH_VARARGS,
    "Execute the code from the given instance of sieve machine "
    "'mach' over the 'message'. " },

  { "set_debug", (PyCFunction) api_sieve_set_debug, METH_VARARGS,
    "" },

  { "set_error", (PyCFunction) api_sieve_set_error, METH_VARARGS,
    "" },

  { "set_parse_error", (PyCFunction) api_sieve_set_parse_error, METH_VARARGS,
    "" },

  { "set_logger", (PyCFunction) api_sieve_set_logger, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_sieve (void)
{
  PySieveMachineType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PySieveMachineType);
}

void
_mu_py_attach_sieve (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PySieveMachineType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PySieveMachineType);
    }
}
