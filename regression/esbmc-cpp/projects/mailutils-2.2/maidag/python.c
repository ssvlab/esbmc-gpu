/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#include "maidag.h"

#ifdef WITH_PYTHON
#include <mailutils/python.h>

int
python_check_msg (mu_message_t msg, struct mu_auth_data *auth,
		  const char *prog)
{
  PyMessage *py_msg;
  mu_py_dict dict[2];
  mu_py_script_data data[1];
  char *argv[] = { "maidag", NULL };

  mu_py_script_init (1, argv);

  if (!log_to_stderr)
    {
      mu_debug_t debug;
      mu_diag_get_debug (&debug);
      mu_py_capture_stderr (debug);
      mu_py_capture_stdout (debug);
    }

  py_msg = PyMessage_NEW ();
  py_msg->msg = msg;
  Py_INCREF (py_msg);

  dict[0].name = "message";
  dict[0].obj  = (PyObject *)py_msg;
  dict[1].name = NULL;
  data[0].module_name = "maidag";
  data[0].attrs = dict;

  mu_py_script_run (prog, data);
  mu_py_script_finish ();
  return 0;
}

#endif /* WITH_PYTHON */

