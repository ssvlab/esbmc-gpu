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
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "pop3d.h"

static struct pop3d_command command_table[] = {
#ifdef WITH_TLS
  { "STLS", pop3d_stls },
# define COMMAND_TABLE_HEAD 1
#else
# define COMMAND_TABLE_HEAD 0
#endif
  { "RETR", pop3d_retr },
  { "DELE", pop3d_dele },
  { "USER", pop3d_user },
  { "QUIT", pop3d_quit },
  { "APOP", pop3d_apop },
  { "AUTH", pop3d_auth },
  { "STAT", pop3d_stat },
  { "LIST", pop3d_list },
  { "NOOP", pop3d_noop },
  { "RSET", pop3d_rset },
  { "TOP",  pop3d_top },
  { "UIDL", pop3d_uidl },
  { "CAPA", pop3d_capa },
  { NULL }
};

static struct pop3d_command *command_table_head =
         command_table + COMMAND_TABLE_HEAD;

pop3d_command_handler_t
pop3d_find_command (const char *name)
{
  struct pop3d_command *p;
  for (p = command_table_head; p->name; p++)
    {
      if (mu_c_strcasecmp (name, p->name) == 0)
	return p->handler;
    }
  return p->handler;
}

#ifdef WITH_TLS
void
enable_stls ()
{
  command_table_head = command_table;
}
#endif

struct error_table
{
  int code;
  const char *text;
};

static struct error_table error_table[] = {
  { ERR_WRONG_STATE, "Incorrect state" },
  { ERR_BAD_ARGS, "Invalid arguments" },
  { ERR_BAD_LOGIN, "Bad login" },
  { ERR_NO_MESG, "No such message" },
  { ERR_MESG_DELE, "Message has been deleted" },
  { ERR_NOT_IMPL, "Not implemented" },
  { ERR_BAD_CMD, "Invalid command" },
  { ERR_MBOX_LOCK, "[IN-USE] Mailbox in use" },
  { ERR_TOO_LONG, "Argument too long" },
  { ERR_NO_MEM, "Out of memory, quitting" },
  { ERR_SIGNAL, "Quitting on signal" },
  { ERR_FILE, "Some deleted messages not removed" },
  { ERR_NO_IFILE, "No input stream" },
  { ERR_NO_OFILE, "No output stream" },
  { ERR_IO, "I/O error" },
  { ERR_PROTO, "Remote protocol error" },
  { ERR_TIMEOUT, "Session timed out" },
  { ERR_UNKNOWN, "Unknown error" },
  { ERR_MBOX_SYNC, "Mailbox was updated by other process" },
#ifdef WITH_TLS
  { ERR_TLS_ACTIVE, "Command not permitted when TLS active" },
#endif /* WITH_TLS */
  { ERR_TLS_IO, "TLS I/O error" },
  { ERR_LOGIN_DELAY,
    "[LOGIN-DELAY] Attempt to log in within the minimum login delay interval" },
  { ERR_TERMINATE, "Terminating on request" },
  { 0 }
};

const char *
pop3d_error_string (int code)
{
  struct error_table *ep;
  for (ep = error_table; ep->code != 0; ep++)
    if (ep->code == code)
      return ep->text;
  return "unknown error";
}
