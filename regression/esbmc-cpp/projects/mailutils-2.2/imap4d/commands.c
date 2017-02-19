/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2003, 2007, 2008, 2010 Free Software
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

struct imap4d_command imap4d_command_table [] =
{
  { "CAPABILITY", imap4d_capability, STATE_ALL, STATE_NONE, STATE_NONE, NULL },
  { "NOOP", imap4d_noop, STATE_ALL, STATE_NONE, STATE_NONE, NULL },
  { "LOGOUT", imap4d_logout, STATE_ALL, STATE_LOGOUT, STATE_NONE, NULL },
  { "AUTHENTICATE", imap4d_authenticate, STATE_NONAUTH, STATE_NONE, STATE_AUTH, NULL },
  { "LOGIN", imap4d_login, STATE_NONAUTH, STATE_NONE, STATE_AUTH, NULL },
  { "SELECT", imap4d_select, STATE_AUTH | STATE_SEL, STATE_AUTH, STATE_SEL, NULL },
  { "EXAMINE", imap4d_examine, STATE_AUTH | STATE_SEL, STATE_AUTH, STATE_SEL, NULL },
  { "CREATE", imap4d_create, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "DELETE", imap4d_delete, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "RENAME", imap4d_rename, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "SUBSCRIBE", imap4d_subscribe, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "UNSUBSCRIBE", imap4d_unsubscribe, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "LIST", imap4d_list, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "LSUB", imap4d_lsub, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "STATUS", imap4d_status, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "APPEND", imap4d_append, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "CHECK", imap4d_check, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "CLOSE", imap4d_close, STATE_SEL, STATE_AUTH, STATE_AUTH, NULL },
  { "UNSELECT", imap4d_unselect, STATE_SEL, STATE_AUTH, STATE_AUTH, NULL },
  { "EXPUNGE", imap4d_expunge, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "SEARCH", imap4d_search, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "FETCH", imap4d_fetch, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "STORE", imap4d_store, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "COPY", imap4d_copy, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "UID", imap4d_uid, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "NAMESPACE", imap4d_namespace, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "ID", imap4d_id, STATE_AUTH | STATE_SEL, STATE_NONE, STATE_NONE, NULL },
  { "IDLE", imap4d_idle, STATE_SEL, STATE_NONE, STATE_NONE, NULL },
#ifdef WITH_TLS
  { "STARTTLS", imap4d_starttls, STATE_NONAUTH, STATE_NONE, STATE_NONE, NULL },
#endif /* WITH_TLS */
  { NULL, 0, 0, 0, 0, NULL }
};
