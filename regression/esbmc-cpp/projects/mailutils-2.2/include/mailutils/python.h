/* GNU Mailutils -- a suite of utilities for electronic mail
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
   Boston, MA 02110-1301 USA */

#ifndef _MAILUTILS_PYTHON_H
#define _MAILUTILS_PYTHON_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  PyObject_HEAD;
  mu_address_t addr;
} PyAddress;

typedef struct
{
  PyObject_HEAD;
  mu_attribute_t attr;
} PyAttribute;

typedef struct
{
  PyObject_HEAD;
  mu_authority_t auth;
} PyAuthority;

typedef struct
{
  PyObject_HEAD;
  mu_ticket_t ticket;
} PyTicket;

typedef struct
{
  PyObject_HEAD;
  mu_wicket_t wicket;
} PyWicket;

typedef struct
{
  PyObject_HEAD;
  struct mu_auth_data *auth_data;
} PyAuthData;

typedef struct
{
  PyObject_HEAD;
  mu_body_t body;
} PyBody;

typedef struct
{
  PyObject_HEAD;
  mu_debug_t dbg;
} PyDebug;

typedef struct
{
  PyObject_HEAD;
  mu_envelope_t env;
} PyEnvelope;

typedef struct
{
  PyObject_HEAD;
  mu_folder_t folder;
} PyFolder;

typedef struct
{
  PyObject_HEAD;
  mu_header_t hdr;
} PyHeader;

typedef struct
{
  PyObject_HEAD;
  mu_mailbox_t mbox;
} PyMailbox;

typedef struct
{
  PyObject_HEAD;
  mu_mailcap_t mc;
} PyMailcap;

typedef struct
{
  PyObject_HEAD;
  mu_mailcap_entry_t entry;
} PyMailcapEntry;

typedef struct
{
  PyObject_HEAD;
  mu_mailer_t mlr;
} PyMailer;

typedef struct
{
  PyObject_HEAD;
  mu_message_t msg;
} PyMessage;

typedef struct
{
  PyObject_HEAD;
  mu_mime_t mime;
} PyMime;

typedef struct
{
  PyObject_HEAD;
  mu_secret_t secret;
} PySecret;

typedef struct
{
  PyObject_HEAD;
  mu_sieve_machine_t mach;
} PySieveMachine;

typedef struct
{
  PyObject_HEAD;
  mu_stream_t stm;
} PyStream;

typedef struct
{
  PyObject_HEAD;
  mu_url_t url;
} PyUrl;

typedef int (*mulist_extractor_fp) (void *data, PyObject **dst);

extern void mu_py_init (void);

extern int mu_py_init_address (void);
extern int mu_py_init_attribute (void);
extern int mu_py_init_auth (void);
extern int mu_py_init_body (void);
extern int mu_py_init_debug (void);
extern int mu_py_init_envelope (void);
extern int mu_py_init_header (void);
extern int mu_py_init_folder (void);
extern int mu_py_init_mailer (void);
extern int mu_py_init_mailbox (void);
extern int mu_py_init_mailcap (void);
extern int mu_py_init_message (void);
extern int mu_py_init_mime (void);
extern int mu_py_init_secret (void);
extern int mu_py_init_sieve (void);
extern int mu_py_init_stream (void);
extern int mu_py_init_url (void);

extern PyObject * mu_py_mulist_to_pylist (mu_list_t list,
					  mulist_extractor_fp fnc);

extern PyAttribute * PyAttribute_NEW ();
extern PyAddress * PyAddress_NEW ();
extern PyAuthority * PyAuthority_NEW ();
extern PyTicket * PyTicket_NEW ();
extern PyWicket * PyWicket_NEW ();
extern PyAuthData * PyAuthData_NEW ();
extern PyBody * PyBody_NEW ();
extern PyDebug * PyDebug_NEW ();
extern PyEnvelope * PyEnvelope_NEW ();
extern PyFolder * PyFolder_NEW ();
extern PyHeader * PyHeader_NEW ();
extern PyMailcap * PyMailcap_NEW ();
extern PyMailcapEntry * PyMailcapEntry_NEW ();
extern PyMailbox * PyMailbox_NEW ();
extern PyMailer * PyMailer_NEW ();
extern PyMessage * PyMessage_NEW ();
extern PyMime * PyMime_NEW ();
extern PySecret * PySecret_NEW ();
extern PyStream * PyStream_NEW ();
extern PyUrl * PyUrl_NEW ();

extern int PyAddress_Check (PyObject *x);
extern int PyAuthority_Check (PyObject *x);
extern int PyTicket_Check (PyObject *x);
extern int PyWicket_Check (PyObject *x);
extern int PyAuthData_Check (PyObject *x);
extern int PyMailbox_Check (PyObject *x);
extern int PyMessage_Check (PyObject *x);
extern int PySecret_Check (PyObject *x);
extern int PyStream_Check (PyObject *x);

typedef struct
{
  char *name;
  PyObject *obj;
} mu_py_dict;

typedef struct
{
  const char *module_name;
  mu_py_dict *attrs;
} mu_py_script_data;

extern void mu_py_script_init (int argc, char *argv[]);
extern void mu_py_script_finish (void);
extern int  mu_py_script_run (const char *filename,
			      mu_py_script_data *data);

extern void mu_py_capture_stdout (mu_debug_t debug);
extern void mu_py_capture_stderr (mu_debug_t debug);

extern int  mu_py_script_process_mailbox (int argc, char *argv[],
					  const char *python_filename,
					  const char *module_name,
					  mu_mailbox_t mbox);

#ifdef __cplusplus
}
#endif

#endif /* _MAILUTILS_PYTHON_H */

