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

#define PY_MODULE "registrar"

struct format_record {
  char *name;
  mu_record_t *record;
};

static struct format_record format_table[] = {
  { "mbox", &mu_mbox_record },
  { "mh", &mu_mh_record },
  { "maildir", &mu_maildir_record },
  { "pop", &mu_pop_record },
  { "imap", &mu_imap_record },
#ifdef WITH_TLS
  { "pops", &mu_pops_record },
  { "imaps", &mu_imaps_record },
#endif /* WITH_TLS */
  { "sendmail", &mu_sendmail_record },
  { "smtp", &mu_smtp_record },
  { NULL, NULL },
};

static mu_record_t *
find_format (const struct format_record *table, const char *name)
{
  for (; table->name; table++)
    if (strcmp (table->name, name) == 0)
      break;
  return table->record;
}

static int
register_format (const char *name)
{
  int status = 0;

  if (!name)
    {
      struct format_record *table;
      for (table = format_table; table->name; table++)
	mu_registrar_record (*table->record);
    }
  else
    {
      mu_record_t *record = find_format (format_table, name);
      if (record)
	status = mu_registrar_record (*record);
      else
	status = EINVAL;
    }
  return status;
}

static int
set_default_format (const char *name)
{
  int status = 0;

  if (name)
    {
      mu_record_t *record = find_format (format_table, name);
      if (record)
	mu_registrar_set_default_record (*record);
      else
	status = EINVAL;
    }
  return status;
}

static PyObject *
api_registrar_register_format (PyObject *self, PyObject *args)
{
  int status;
  char *name = NULL;

  if (!PyArg_ParseTuple (args, "|s", &name))
    return NULL;

  status = register_format (name);
  return _ro (PyInt_FromLong (status));
}

static PyObject *
api_registrar_set_default_format (PyObject *self, PyObject *args)
{
  int status;
  char *name = NULL;

  if (!PyArg_ParseTuple (args, "s", &name))
    return NULL;

  status = set_default_format (name);
  return _ro (PyInt_FromLong (status));
}

static PyMethodDef methods[] = {
  { "register_format", (PyCFunction) api_registrar_register_format,
    METH_VARARGS,
    "Register desired mailutils formats. Any number of arguments "
    "can be given." },

  { "set_default_format", (PyCFunction) api_registrar_set_default_format,
    METH_VARARGS, "" },

  { NULL, NULL, 0, NULL }
};

void
_mu_py_attach_registrar ()
{
  _mu_py_attach_module (PY_MODULE, methods);

  mu_registrar_record (MU_DEFAULT_RECORD);
  mu_registrar_set_default_record (MU_DEFAULT_RECORD);

#ifdef WITH_TLS
  mu_init_tls_libs ();
#endif /* WITH_TLS */

}
