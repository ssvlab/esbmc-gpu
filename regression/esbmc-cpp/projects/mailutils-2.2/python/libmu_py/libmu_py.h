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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <mailutils/address.h>
#include <mailutils/attribute.h>
#include <mailutils/auth.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/envelope.h>
#include <mailutils/error.h>
#include <mailutils/filter.h>
#include <mailutils/folder.h>
#include <mailutils/header.h>
#include <mailutils/list.h>
#include <mailutils/mailbox.h>
#include <mailutils/mailcap.h>
#include <mailutils/mailer.h>
#include <mailutils/message.h>
#include <mailutils/mime.h>
#include <mailutils/mu_auth.h>
#include <mailutils/mutil.h>
#include <mailutils/nls.h>
#include <mailutils/registrar.h>
#include <mailutils/tls.h>
#include <mailutils/secret.h>
#include <mailutils/sieve.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/python.h>

#define PY_PACKAGE_NAME "mailutils"
#define PY_ROOT_NAME "c_api"
#define PY_PACKAGE_VERSION PACKAGE_VERSION

extern inline PyObject * _ro (PyObject *obj);
extern void _py_dealloc (PyObject *self);
extern PyObject * status_object (int status, PyObject *py_obj);
extern PyObject * _mu_py_attach_module (const char *name, PyMethodDef *methods);

extern void mu_py_attach_modules (void);

extern void _mu_py_attach_error (void);
extern void _mu_py_attach_address (void);
extern void _mu_py_attach_attribute (void);
extern void _mu_py_attach_auth (void);
extern void _mu_py_attach_body (void);
extern void _mu_py_attach_debug (void);
extern void _mu_py_attach_envelope (void);
extern void _mu_py_attach_header  (void);
extern void _mu_py_attach_filter (void);
extern void _mu_py_attach_folder (void);
extern void _mu_py_attach_mailer (void);
extern void _mu_py_attach_mailbox (void);
extern void _mu_py_attach_mailcap (void);
extern void _mu_py_attach_message (void);
extern void _mu_py_attach_mime (void);
extern void _mu_py_attach_nls (void);
extern void _mu_py_attach_registrar (void);
extern void _mu_py_attach_secret (void);
extern void _mu_py_attach_sieve (void);
extern void _mu_py_attach_stream (void);
extern void _mu_py_attach_url (void);
extern void _mu_py_attach_util (void);
