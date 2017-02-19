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
#include <mailutils/pam.h>

#define PY_MODULE  "auth"
#define PY_CSNAME1 "AuthorityType"
#define PY_CSNAME2 "TicketType"
#define PY_CSNAME3 "WicketType"
#define PY_CSNAME4 "AuthDataType"

static PyObject *
_repr1 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME1 " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyObject *
_repr2 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME2 " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyObject *
_repr3 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME3 " instance at %p>", self);
  return PyString_FromString (buf);
}

static PyObject *
_repr4 (PyObject *self)
{
  char buf[80];
  sprintf (buf, "<" PY_MODULE "." PY_CSNAME4 " instance at %p>", self);
  return PyString_FromString (buf);
}


static PyTypeObject PyAuthorityType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME1,  /* tp_name */
  sizeof (PyAuthority),      /* tp_basicsize */
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

static PyTypeObject PyTicketType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME2,  /* tp_name */
  sizeof (PyTicket),         /* tp_basicsize */
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

static PyTypeObject PyWicketType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME3,  /* tp_name */
  sizeof (PyWicket),         /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)_py_dealloc,   /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr; __getattr__ */
  0,                         /* tp_setattr; __setattr__ */
  0,                         /* tp_compare; __cmp__ */
  _repr3,                    /* tp_repr; __repr__ */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash; __hash__ */
  0,                         /* tp_call; __call__ */
  _repr3,                    /* tp_str; __str__ */
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

void
_dealloc4 (PyObject *self)
{
  PyAuthData *py_ad = (PyAuthData *)self;
  if (py_ad->auth_data)
    mu_auth_data_free (py_ad->auth_data);
  self->ob_type->tp_free (self);
}

static PyObject *
_getattr4 (PyObject *self, char *name)
{
  PyAuthData *py_ad = (PyAuthData *)self;
  struct mu_auth_data *ad = py_ad->auth_data;

  if (!ad)
    return NULL;

  if (strcmp (name, "name") == 0) {
    return PyString_FromString (ad->name);
  }
  else if (strcmp (name, "passwd") == 0) {
    return PyString_FromString (ad->passwd);
  }
  else if (strcmp (name, "uid") == 0) {
    return PyInt_FromLong (ad->uid);
  }
  else if (strcmp (name, "gid") == 0) {
    return PyInt_FromLong (ad->gid);
  }
  else if (strcmp (name, "gecos") == 0) {
    return PyString_FromString (ad->gecos);
  }
  else if (strcmp (name, "dir") == 0) {
    return PyString_FromString (ad->dir);
  }
  else if (strcmp (name, "shell") == 0) {
    return PyString_FromString (ad->shell);
  }
  else if (strcmp (name, "mailbox") == 0) {
    return PyString_FromString (ad->mailbox);
  }
  else if (strcmp (name, "source") == 0) {
    return PyString_FromString (ad->source);
  }
  else if (strcmp (name, "quota") == 0) {
    return PyInt_FromLong (ad->quota);
  }
  else if (strcmp (name, "flags") == 0) {
    return PyInt_FromLong (ad->flags);
  }
  else if (strcmp (name, "change_uid") == 0) {
    return PyInt_FromLong (ad->change_uid);
  }
  return NULL;
}

static PyTypeObject PyAuthDataType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  PY_MODULE "." PY_CSNAME4,  /* tp_name */
  sizeof (PyAuthData),       /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)_dealloc4,     /* tp_dealloc */
  0,                         /* tp_print */
  _getattr4,                 /* tp_getattr; __getattr__ */
  0,                         /* tp_setattr; __setattr__ */
  0,                         /* tp_compare; __cmp__ */
  _repr4,                    /* tp_repr; __repr__ */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash; __hash__ */
  0,                         /* tp_call; __call__ */
  _repr4,                    /* tp_str; __str__ */
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

PyAuthority *
PyAuthority_NEW ()
{
  return (PyAuthority *)PyObject_NEW (PyAuthority, &PyAuthorityType);
}

int
PyAuthority_Check (PyObject *x)
{
  return x->ob_type == &PyAuthorityType;
}

PyTicket *
PyTicket_NEW ()
{
  return (PyTicket *)PyObject_NEW (PyTicket, &PyTicketType);
}

int
PyTicket_Check (PyObject *x)
{
  return x->ob_type == &PyTicketType;
}

PyWicket *
PyWicket_NEW ()
{
  return (PyWicket *)PyObject_NEW (PyWicket, &PyWicketType);
}

int
PyWicket_Check (PyObject *x)
{
  return x->ob_type == &PyWicketType;
}

PyAuthData *
PyAuthData_NEW ()
{
  return (PyAuthData *)PyObject_NEW (PyAuthData, &PyAuthDataType);
}

int
PyAuthData_Check (PyObject *x)
{
  return x->ob_type == &PyAuthDataType;
}

/*
 *  Authority
 */

static PyObject *
api_authority_create (PyObject *self, PyObject *args)
{
  int status;
  PyAuthority *py_auth;
  PyTicket *py_ticket;

  if (!PyArg_ParseTuple (args, "O!O!",
			 &PyAuthorityType, &py_auth,
			 &PyTicketType, &py_ticket))
    return NULL;

  status = mu_authority_create (&py_auth->auth, py_ticket->ticket, NULL);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_authority_destroy (PyObject *self, PyObject *args)
{
  PyAuthority *py_auth;

  if (!PyArg_ParseTuple (args, "O!", &PyAuthorityType, &py_auth))
    return NULL;

  mu_authority_destroy (&py_auth->auth, NULL);
  return _ro (Py_None);
}

static PyObject *
api_authority_get_ticket (PyObject *self, PyObject *args)
{
  int status;
  PyAuthority *py_auth;
  PyTicket *py_ticket = PyTicket_NEW ();

  if (!PyArg_ParseTuple (args, "O!", &PyAuthorityType, &py_auth))
    return NULL;

  Py_INCREF (py_ticket);

  status = mu_authority_get_ticket (py_auth->auth, &py_ticket->ticket);
  return status_object (status, (PyObject *)py_ticket);
}

static PyObject *
api_authority_set_ticket (PyObject *self, PyObject *args)
{
  int status;
  PyAuthority *py_auth;
  PyTicket *py_ticket;

  if (!PyArg_ParseTuple (args, "O!O!",
			 &PyAuthorityType, &py_auth,
			 &PyTicketType, &py_ticket))
    return NULL;

  status = mu_authority_set_ticket (py_auth->auth, py_ticket->ticket);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_authority_authenticate (PyObject *self, PyObject *args)
{
  int status;
  PyAuthority *py_auth;

  if (!PyArg_ParseTuple (args, "O!", &PyAuthorityType, &py_auth))
    return NULL;

  status = mu_authority_authenticate (py_auth->auth);
  return _ro (PyInt_FromLong (status));
}

/*
 *  Ticket
 */

static PyObject *
api_ticket_create (PyObject *self, PyObject *args)
{
  int status;
  PyTicket *py_ticket;

  if (!PyArg_ParseTuple (args, "O!", &PyTicketType, &py_ticket))
    return NULL;

  status = mu_ticket_create (&py_ticket->ticket, NULL);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_ticket_destroy (PyObject *self, PyObject *args)
{
  PyTicket *py_ticket;

  if (!PyArg_ParseTuple (args, "O!", &PyTicketType, &py_ticket))
    return NULL;

  mu_ticket_destroy (&py_ticket->ticket);
  return _ro (Py_None);
}

static PyObject *
api_ticket_set_secret (PyObject *self, PyObject *args)
{
  int status;
  PyTicket *py_ticket;
  PySecret *py_secret;

  if (!PyArg_ParseTuple (args, "O!O", &PyTicketType, &py_ticket, &py_secret))
    return NULL;

  if (!PySecret_Check ((PyObject *)py_secret))
    {
      PyErr_SetString (PyExc_TypeError, "");
      return NULL;
    }

  status = mu_ticket_set_secret (py_ticket->ticket,
				 py_secret->secret);
  return _ro (PyInt_FromLong (status));
}

/*
 *  Wicket
 */

static PyObject *
api_wicket_create (PyObject *self, PyObject *args)
{
  int status;
  char *filename;
  PyWicket *py_wicket;

  if (!PyArg_ParseTuple (args, "O!s", &PyWicketType, &py_wicket, &filename))
    return NULL;

  status = mu_file_wicket_create (&py_wicket->wicket, filename);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_wicket_destroy (PyObject *self, PyObject *args)
{
  PyWicket *py_wicket;

  if (!PyArg_ParseTuple (args, "O!", &PyWicketType, &py_wicket))
    return NULL;

  mu_wicket_destroy (&py_wicket->wicket);
  return _ro (Py_None);
}

static PyObject *
api_wicket_get_ticket (PyObject *self, PyObject *args)
{
  int status;
  char *user;
  PyWicket *py_wicket;
  PyTicket *py_ticket = PyTicket_NEW ();

  if (!PyArg_ParseTuple (args, "O!s", &PyWicketType, &py_wicket, &user))
    return NULL;

  Py_INCREF (py_ticket);

  status = mu_wicket_get_ticket (py_wicket->wicket, user,
				 &py_ticket->ticket);
  return status_object (status, (PyObject *)py_ticket);
}

/*
 *  mu_auth
 */

struct module_record {
  char *name;
  struct mu_auth_module *module;
};

static struct module_record module_table[] = {
  { "system",  &mu_auth_system_module },
  { "generic", &mu_auth_generic_module },
  { "pam",     &mu_auth_pam_module },
  { "sql",     &mu_auth_sql_module },
  { "virtual", &mu_auth_virtual_module },
  { "radius",  &mu_auth_radius_module },
  { "ldap",    &mu_auth_ldap_module },
  { NULL, NULL },
};

static struct mu_auth_module *
find_module (const struct module_record *table, const char *name)
{
  for (; table->name; table++)
    if (strcmp (table->name, name) == 0)
      break;
  return table->module;
}

static int
register_module (const char *name)
{
  int status = 0;

  if (!name)
    {
      struct module_record *table;
      for (table = module_table; table->name; table++)
	mu_auth_register_module (table->module);
    }
  else
    {
      struct mu_auth_module *module = find_module (module_table, name);
      if (module)
	mu_auth_register_module (module);
      else
	status = EINVAL;
    }
  return status;
}

static PyObject *
api_register_module (PyObject *self, PyObject *args)
{
  int status;
  char *name = NULL;

  if (!PyArg_ParseTuple (args, "|s", &name))
    return NULL;

  status = register_module (name);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_get_auth_by_name (PyObject *self, PyObject *args)
{
  char *username;
  PyAuthData *py_ad = PyAuthData_NEW ();

  if (!PyArg_ParseTuple (args, "s", &username))
    return NULL;

  Py_INCREF (py_ad);

  py_ad->auth_data = mu_get_auth_by_name (username);
  if (!py_ad->auth_data)
    return _ro (Py_None);

  return _ro ((PyObject *)py_ad);
}

static PyObject *
api_get_auth_by_uid (PyObject *self, PyObject *args)
{
  uid_t uid;
  PyAuthData *py_ad = PyAuthData_NEW ();

  if (!PyArg_ParseTuple (args, "i", &uid))
    return NULL;

  Py_INCREF (py_ad);

  py_ad->auth_data = mu_get_auth_by_uid (uid);
  if (!py_ad->auth_data)
    return _ro (Py_None);

  return _ro ((PyObject *)py_ad);
}

static PyObject *
api_authenticate (PyObject *self, PyObject *args)
{
  int status;
  char *pass;
  PyAuthData *py_ad;

  if (!PyArg_ParseTuple (args, "O!s", &PyAuthDataType, &py_ad, &pass))
    return NULL;

  status = mu_authenticate (py_ad->auth_data, pass);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_set_pam_service (PyObject *self, PyObject *args)
{
  char *pam_service;

  if (!PyArg_ParseTuple (args, "s", &pam_service))
    return NULL;

  mu_pam_service = pam_service;
  return _ro (Py_None);
}

static PyMethodDef methods[] = {
  { "authority_create", (PyCFunction) api_authority_create, METH_VARARGS,
    "" },

  { "authority_destroy", (PyCFunction) api_authority_destroy, METH_VARARGS,
    "" },

  { "authority_get_ticket", (PyCFunction) api_authority_get_ticket,
    METH_VARARGS, "" },

  { "authority_set_ticket", (PyCFunction) api_authority_set_ticket,
    METH_VARARGS, "" },

  { "authority_authenticate", (PyCFunction) api_authority_authenticate,
    METH_VARARGS, "" },

  { "ticket_create", (PyCFunction) api_ticket_create, METH_VARARGS,
    "" },

  { "ticket_destroy", (PyCFunction) api_ticket_destroy, METH_VARARGS,
    "" },

  { "ticket_set_secret", (PyCFunction) api_ticket_set_secret, METH_VARARGS,
    "" },

  { "wicket_create", (PyCFunction) api_wicket_create, METH_VARARGS,
    "" },

  { "wicket_destroy", (PyCFunction) api_wicket_destroy, METH_VARARGS,
    "" },

  { "wicket_get_ticket", (PyCFunction) api_wicket_get_ticket, METH_VARARGS,
    "" },

  { "register_module", (PyCFunction) api_register_module, METH_VARARGS,
    "" },

  { "get_auth_by_name", (PyCFunction) api_get_auth_by_name, METH_VARARGS,
    "" },

  { "get_auth_by_uid", (PyCFunction) api_get_auth_by_uid, METH_VARARGS,
    "" },

  { "authenticate", (PyCFunction) api_authenticate, METH_VARARGS,
    "" },

  { "set_pam_service", (PyCFunction) api_set_pam_service, METH_VARARGS,
    "" },

  { NULL, NULL, 0, NULL }
};

int
mu_py_init_auth (void)
{
  PyAuthorityType.tp_new = PyType_GenericNew;
  PyTicketType.tp_new = PyType_GenericNew;
  PyWicketType.tp_new = PyType_GenericNew;
  PyAuthDataType.tp_new = PyType_GenericNew;

  if (PyType_Ready (&PyAuthorityType) < 0)
    return -1;
  if (PyType_Ready (&PyTicketType) < 0)
    return -1;
  if (PyType_Ready (&PyWicketType) < 0)
    return -1;
  if (PyType_Ready (&PyAuthDataType) < 0)
    return -1;
  return 0;
}

void
_mu_py_attach_auth (void)
{
  PyObject *m;
  if ((m = _mu_py_attach_module (PY_MODULE, methods)))
    {
      Py_INCREF (&PyAuthorityType);
      Py_INCREF (&PyTicketType);
      Py_INCREF (&PyWicketType);
      Py_INCREF (&PyAuthDataType);

      PyModule_AddObject (m, PY_CSNAME1, (PyObject *)&PyAuthorityType);
      PyModule_AddObject (m, PY_CSNAME2, (PyObject *)&PyTicketType);
      PyModule_AddObject (m, PY_CSNAME3, (PyObject *)&PyWicketType);
      PyModule_AddObject (m, PY_CSNAME4, (PyObject *)&PyAuthDataType);
    }
}
