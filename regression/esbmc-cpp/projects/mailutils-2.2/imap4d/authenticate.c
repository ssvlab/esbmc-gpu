/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2006, 2007, 2008, 2010 Free Software
   Foundation, Inc.

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

#include "imap4d.h"

struct imap_auth {
  char *name;
  imap4d_auth_handler_fp handler;
};

static mu_list_t imap_auth_list;

static int
comp (const void *item, const void *data)
{
  const struct imap_auth *p = item;
  return strcmp (p->name, (const char*) data);
}

void
auth_add (char *name, imap4d_auth_handler_fp handler)
{
  struct imap_auth *p = malloc (sizeof (*p));

  if (!p)
    imap4d_bye (ERR_NO_MEM);

  p->name = name;
  p->handler = handler;
  if (!imap_auth_list)
    {
      mu_list_create (&imap_auth_list);
      mu_list_set_comparator (imap_auth_list, comp);
      mu_list_set_destroy_item (imap_auth_list, mu_list_free_item);
    }
  mu_list_append (imap_auth_list, (void*)p);
}

void
auth_remove (char *name)
{
  mu_list_remove (imap_auth_list, (void*) name);
}

static int
_auth_capa (void *item, void *usused)
{
  struct imap_auth *p = item;
  util_send(" AUTH=%s", p->name);
  return 0;
}

struct auth_data {
  struct imap4d_command *command;
  char *auth_type;
  char *arg;
  char *username;
  int result;
};

static int
_auth_try (void *item, void *data)
{
  struct imap_auth *p = item;
  struct auth_data *ap = data;

  if (strcmp (p->name, ap->auth_type) == 0)
    {
      ap->result = p->handler (ap->command, ap->auth_type, &ap->username);
      return 1;
    }
  return 0;
}

void
imap4d_auth_capability ()
{
  mu_list_do (imap_auth_list, _auth_capa, NULL);
}

/*
6.2.1.  AUTHENTICATE Command

   Arguments:  authentication mechanism name
*/

int
imap4d_authenticate (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  char *auth_type;
  struct auth_data adata;

  if (imap4d_tokbuf_argc (tok) != 3)
    return util_finish (command, RESP_BAD, "Invalid arguments");
  
  auth_type = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);

  if (tls_required)
    return util_finish (command, RESP_NO,
			"Command disabled: Use STARTTLS first");
  
  adata.command = command;
  adata.auth_type = auth_type;
  adata.arg = NULL;
  adata.username = NULL;

  if (mu_list_do (imap_auth_list, _auth_try, &adata) == 0)
    return util_finish (command, RESP_NO,
			"Authentication mechanism not supported");
  
  if (adata.result == RESP_OK && adata.username)
    {
      if (imap4d_session_setup (adata.username))
	return util_finish (command, RESP_NO,
			    "User name or passwd rejected");
      else
	return util_finish (command, RESP_OK,
			    "%s authentication successful", auth_type);
    }
      
  return util_finish (command, adata.result,
		      "%s authentication failed", auth_type);
}

