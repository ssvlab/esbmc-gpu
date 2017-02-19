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

#define PY_MODULE "debug"
#define PY_CSNAME "DebugType"

static PyObject *
_repr (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyTypeObject PyDebugType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME,   /* tp_name */
  sizeof (PyDebug),          /* tp_basicsize */
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

PyDebug *
PyDebug_NEW ()
{
  return (PyDebug *)PyObject_NEW (PyDebug, &PyDebugType);
}

static PyObject *
api_debug_set_level (PyObject *self, PyObject *args)
{
  int status, level;
  PyDebug *py_dbg;

  if (!PyArg_ParseTuple (args, "O!i", &PyDebugType, &py_dbg, &level))
    return NULL;

  status = mu_debug_set_level (py_dbg->dbg,
			       MU_DEBUG_LEVEL_UPTO (level));
  return _ro (PyInt_FromLong (status));
}

static PyMethodDef methods[] = {
  { "set_level", (PyCFunction) api_debug_set_level, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

static mu_debug_t _mu_prog_debug_stdout;
static mu_debug_t _mu_prog_debug_stderr;

static PyObject *
_capture_stdout (PyObject *self, PyObject *args)
{
  char *buf = "";
  if (!PyArg_ParseTuple (args, "s", &buf))
    return NULL;
  if (_mu_prog_debug_stdout)
    mu_debug_printf (_mu_prog_debug_stdout, MU_DIAG_INFO, "%.*s",
		     (int) strlen (buf), buf);
  return _ro (Py_None);
}

static PyObject *
_capture_stderr (PyObject *self, PyObject *args)
{
  char *buf = "";
  if (!PyArg_ParseTuple (args, "s", &buf))
    return NULL;
  if (_mu_prog_debug_stderr)
    mu_debug_printf (_mu_prog_debug_stderr, MU_DIAG_ERROR, "%.*s",
		     (int) strlen (buf), buf);
  return _ro (Py_None);
}

static PyMethodDef capture_stdout_method[] =
{
  { "write", _capture_stdout, 1 },
  { NULL, NULL, 0, NULL }
};

static PyMethodDef capture_stderr_method[] =
{
  { "write", _capture_stderr, 1 },
  { NULL, NULL, 0, NULL }
};

void
mu_py_capture_stdout (mu_debug_t debug)
{
  PyObject *py_out;
  _mu_prog_debug_stdout = debug;
  py_out = Py_InitModule ("stdout", capture_stdout_method);
  if (py_out)
    PySys_SetObject ("stdout", py_out);
}

void
mu_py_capture_stderr (mu_debug_t debug)
{
  PyObject *py_err;
  _mu_prog_debug_stderr = debug;
  py_err = Py_InitModule ("stderr", capture_stderr_method);
  if (py_err)
    PySys_SetObject ("stderr", py_err);
}

int
mu_py_init_debug (void)
{
  PyDebugType.tp_new = PyType_GenericNew;
  return PyType_Ready (&PyDebugType);
}

void
_mu_py_attach_debug (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyDebugType);
      PyModule_AddObject (m, PY_CSNAME, (PyObject *)&PyDebugType);
    }
}
