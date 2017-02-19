/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2006, 2007, 2008, 2009, 2010 Free
   Software Foundation, Inc.

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

static int
_append_date (mu_envelope_t envelope, char *buf, size_t len, size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  size_t size;
  if (!buf)
    size = MU_ENVELOPE_DATE_LENGTH;
  else
    {
      struct tm **tm = mu_message_get_owner (msg);
      size = mu_strftime (buf, len, "%a %b %d %H:%M:%S %Y", *tm);
    }
  if (pnwrite)
    *pnwrite = size;
  return 0;
}

static int
_append_sender (mu_envelope_t envelope, char *buf, size_t len, size_t *pnwrite)
{
  size_t n = mu_cpystr (buf, "GNU-imap4d", len);
  if (pnwrite)
    *pnwrite = n;
  return 0;
}

static int
_append_size (mu_message_t msg, size_t *psize)
{
  mu_stream_t str;
  int status = mu_message_get_stream (msg, &str);
  if (status == 0)
    {
      mu_off_t size;
      status = mu_stream_size (str, &size);
      if (status == 0 && psize)
	*psize = size;
    }
  return status;
}

int
imap4d_append0 (mu_mailbox_t mbox, int flags, char *date_time, char *text,
		char **err_text)
{
  mu_stream_t stream;
  int rc = 0;
  size_t len = 0;
  mu_message_t msg = 0;
  struct tm *tm;
  time_t t;
  mu_envelope_t env;
    
  if (mu_message_create (&msg, &tm))
    return 1;
  
  if (mu_memory_stream_create (&stream, 0, MU_STREAM_RDWR)
      || mu_stream_open (stream))
    {
      mu_message_destroy (&msg, &tm);
      return 1;
    }

  /* If a date_time is specified, the internal date SHOULD be set in the
     resulting message; otherwise, the internal date of the resulting
     message is set to the current date and time by default. */
  if (date_time)
    {
      if (util_parse_internal_date (date_time, &t))
	{
	  *err_text = "Invalid date/time format";
	  return 1;
	}
    }
  else
    time(&t);
  
  tm = gmtime(&t);

  while (*text && mu_isblank (*text))
    text++;

  mu_stream_write (stream, text, strlen (text), len, &len);
  mu_message_set_stream (msg, stream, &tm);
  mu_message_set_size (msg, _append_size, &tm);

  mu_envelope_create (&env, msg);
  mu_envelope_set_date (env, _append_date, msg);
  mu_envelope_set_sender (env, _append_sender, msg);
  mu_message_set_envelope (msg, env, &tm);
  rc = mu_mailbox_append_message (mbox, msg);
  if (rc == 0 && flags)
    {
      size_t num = 0;
      mu_attribute_t attr = NULL;
      mu_mailbox_messages_count (mbox, &num);
      mu_mailbox_get_message (mbox, num, &msg);
      mu_message_get_attribute (msg, &attr);
      mu_attribute_set_flags (attr, flags);
    }

  mu_message_destroy (&msg, &tm);
  return rc;
}


/* APPEND mbox [(flags)] [date_time] message_literal */
int
imap4d_append (struct imap4d_command *command, imap4d_tokbuf_t tok)
{
  int i;
  char *mboxname;
  int flags = 0;
  mu_mailbox_t dest_mbox = NULL;
  int status;
  int argc = imap4d_tokbuf_argc (tok);
  char *date_time;
  char *msg_text;
  char *err_text = "[TRYCREATE] failed";
  
  if (argc < 4)
    return util_finish (command, RESP_BAD, "Too few arguments");
      
  mboxname = imap4d_tokbuf_getarg (tok, IMAP4_ARG_1);
  if (!mboxname)
    return util_finish (command, RESP_BAD, "Too few arguments");

  i = IMAP4_ARG_2;
  if (imap4d_tokbuf_getarg (tok, i)[0] == '(')
    {
      while (++i < argc)
	{
	  int type;
	  char *arg = imap4d_tokbuf_getarg (tok, i);
	  
	  if (!util_attribute_to_type (arg, &type))
	    flags |= type;
	  else if (arg[0] == ')')
	    break;
	}
      if (i == argc)
	return util_finish (command, RESP_BAD, "Missing closing parenthesis");
      i++;
    }

  switch (argc - i)
    {
    case 2:
      /* Date/time is present */
      date_time = imap4d_tokbuf_getarg (tok, i);
      i++;
      break;

    case 1:
      date_time = NULL;
      break;

    default:
      return util_finish (command, RESP_BAD, "Too many arguments");
    }

  msg_text = imap4d_tokbuf_getarg (tok, i);
  
  mboxname = namespace_getfullpath (mboxname, "/", NULL);
  if (!mboxname)
    return util_finish (command, RESP_NO, "Couldn't open mailbox"); 

  status = mu_mailbox_create_default (&dest_mbox, mboxname);
  if (status == 0)
    {
      /* It SHOULD NOT automatically create the mailbox. */
      status = mu_mailbox_open (dest_mbox, MU_STREAM_RDWR);
      if (status == 0)
	{
	  status = imap4d_append0 (dest_mbox, flags, date_time, msg_text,
				   &err_text);
	  mu_mailbox_close (dest_mbox);
	}
      mu_mailbox_destroy (&dest_mbox);
    }
  
  free (mboxname);
  if (status == 0)
    return util_finish (command, RESP_OK, "Completed");

  return util_finish (command, RESP_NO, err_text);
}


