/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2006, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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

#include "mu_scm.h"

static scm_t_bits message_tag;

struct mu_message
{
  mu_message_t msg;       /* Message itself */
  SCM mbox;               /* Mailbox it belongs to */
  int needs_destroy;      /* Set during mark phase if the message needs
			     explicit destroying */
};

/* SMOB functions: */

static SCM
mu_scm_message_mark (SCM message_smob)
{
  struct mu_message *mum = (struct mu_message *) SCM_CDR (message_smob);
  if (mu_message_get_owner (mum->msg) == NULL)
    mum->needs_destroy = 1;
  return mum->mbox;
}

static scm_sizet
mu_scm_message_free (SCM message_smob)
{
  struct mu_message *mum = (struct mu_message *) SCM_CDR (message_smob);
  if (mum->needs_destroy)
    mu_message_destroy (&mum->msg, NULL);
  free (mum);
  return 0;
}

static char *
_get_envelope_sender (mu_envelope_t env)
{
  mu_address_t addr;
  const char *buffer;
  char *ptr;
  
  if (mu_envelope_sget_sender (env, &buffer)
      || mu_address_create (&addr, buffer))
    return NULL;

  if (mu_address_aget_email (addr, 1, &ptr))
    {
      mu_address_destroy (&addr);
      return NULL;
    }

  mu_address_destroy (&addr);
  return ptr;
}

static int
mu_scm_message_print (SCM message_smob, SCM port, scm_print_state * pstate)
{
  struct mu_message *mum = (struct mu_message *) SCM_CDR (message_smob);
  mu_envelope_t env = NULL;
  const char *buffer;
  const char *p;
  size_t m_size = 0, m_lines = 0;
  struct tm tm;
  mu_timezone tz;
  char datebuf[sizeof ("Mon Jan 01 00:00")]; /* Warning: length must be > 9 */

  mu_message_get_envelope (mum->msg, &env);

  scm_puts ("#<message ", port);

  if (message_smob == SCM_BOOL_F)
    {
      /* several mu_message.* functions may return #f */
      scm_puts ("#f", port);
    }
  else
    {
      p = _get_envelope_sender (env);
      scm_puts ("\"", port);
      if (p)
	{
	  scm_puts (p, port);
	  free ((void *) p);
	}
      else
	scm_puts ("UNKNOWN", port);
      
      if (mu_envelope_sget_date (env, &p) == 0
          && mu_parse_ctime_date_time (&p, &tm, &tz) == 0)
	{
	  strftime (datebuf, sizeof (datebuf), "%a %b %e %H:%M", &tm);
	  buffer = datebuf;
	}
      else
	buffer = "UNKNOWN";
      scm_puts ("\" \"", port);
      scm_puts (buffer, port);
      scm_puts ("\" ", port);
      
      mu_message_size (mum->msg, &m_size);
      mu_message_lines (mum->msg, &m_lines);
      
      snprintf (datebuf, sizeof (datebuf), "%3lu %-5lu",
		(unsigned long) m_lines, (unsigned long) m_size);
      scm_puts (datebuf, port);
    }
  scm_puts (">", port);
  return 1;
}

/* Internal calls: */

SCM
mu_scm_message_create (SCM owner, mu_message_t msg)
{
  struct mu_message *mum;

  mum = scm_gc_malloc (sizeof (struct mu_message), "message");
  mum->msg = msg;
  mum->mbox = owner;
  mum->needs_destroy = 0;
  SCM_RETURN_NEWSMOB (message_tag, mum);
}

void
mu_scm_message_add_owner (SCM MESG, SCM owner)
{
  struct mu_message *mum = (struct mu_message *) SCM_CDR (MESG);
  SCM cell;

  if (scm_is_bool (mum->mbox))
    {
      mum->mbox = owner;
      return;
    }
  
  if (scm_is_pair (mum->mbox))
    cell = scm_cons (owner, mum->mbox);
  else
    cell = scm_cons (owner, scm_cons (mum->mbox, SCM_EOL));
  mum->mbox = cell;
}

mu_message_t
mu_scm_message_get (SCM MESG)
{
  struct mu_message *mum = (struct mu_message *) SCM_CDR (MESG);
  return mum->msg;
}

int
mu_scm_is_message (SCM scm)
{
  return SCM_NIMP (scm) && (long) SCM_CAR (scm) == message_tag;
}

/* ************************************************************************* */
/* Guile primitives */

SCM_DEFINE_PUBLIC (scm_mu_message_create, "mu-message-create", 0, 0, 0,
		   (),
		   "Creates an empty message.\n")
#define FUNC_NAME s_scm_mu_message_create
{
  mu_message_t msg;
  mu_message_create (&msg, NULL);
  return mu_scm_message_create (SCM_BOOL_F, msg);
}
#undef FUNC_NAME

/* FIXME: This changes envelope date */
SCM_DEFINE_PUBLIC (scm_mu_message_copy, "mu-message-copy", 1, 0, 0,
		   (SCM mesg),
		   "Creates a copy of the message @var{mesg}.\n")
#define FUNC_NAME s_scm_mu_message_copy
{
  mu_message_t msg, newmsg;
  mu_stream_t in = NULL, out = NULL;
  char buffer[512];
  size_t off, n;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);

  status = mu_message_get_stream (msg, &in);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get input stream from message ~A",
		  scm_list_1 (mesg));
  
  status = mu_message_create (&newmsg, NULL);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot create message", SCM_BOOL_F);
  
  status = mu_message_get_stream (newmsg, &out);
  if (status)
    {
      mu_message_destroy (&newmsg, NULL);
      mu_scm_error (FUNC_NAME, status,
		    "Cannot get output stream", SCM_BOOL_F);
    }

  off = 0;
  while ((status = mu_stream_read (in, buffer, sizeof (buffer) - 1, off, &n))
	 == 0
	 && n != 0)
    {
      size_t wr;
      int rc;
      
      rc = mu_stream_write (out, buffer, n, off, &wr);
      if (rc)
	{
	  mu_message_destroy (&newmsg, NULL);
	  mu_scm_error (FUNC_NAME, rc, "Error writing to stream", SCM_BOOL_F);
	}
      
      off += n;
      if (wr != n)
	{
	  mu_message_destroy (&newmsg, NULL);
	  mu_scm_error (FUNC_NAME, rc, "Error writing to stream: Short write",
			SCM_BOOL_F);
	}
    }
  
  return mu_scm_message_create (SCM_BOOL_F, newmsg);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_destroy, "mu-message-destroy", 1, 0, 0,
		   (SCM mesg),
		   "Destroys the message @var{mesg}.")
#define FUNC_NAME s_scm_mu_message_destroy
{
  struct mu_message *mum;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  mum = (struct mu_message *) SCM_CDR (mesg);
  mu_message_destroy (&mum->msg, mu_message_get_owner (mum->msg));
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_set_header, "mu-message-set-header", 3, 1, 0,
		   (SCM mesg, SCM header, SCM value, SCM replace),
"Sets header @var{header} of the message @var{mesg} to new @var{value}.\n"
"If @var{header} is already present in the message, its value\n"
"is replaced with the suplied one iff the optional @var{replace} is\n"
"@code{#t}. Otherwise, a new header is created and appended.")
#define FUNC_NAME s_scm_mu_message_set_header
{
  mu_message_t msg;
  mu_header_t hdr;
  int repl = 0;
  int status;
  char *hdr_c, *val_c;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_string (header), header, SCM_ARG2, FUNC_NAME);

  if (scm_is_bool (value))
    return SCM_UNSPECIFIED;/*FIXME: Exception*/
  
  SCM_ASSERT (scm_is_string (value), value, SCM_ARG3, FUNC_NAME);
  if (!SCM_UNBNDP (replace))
    {
      repl = replace == SCM_BOOL_T;
    }
  
  status = mu_message_get_header (msg, &hdr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message headers", SCM_BOOL_F);

  hdr_c = scm_to_locale_string (header);
  val_c = scm_to_locale_string (value);
  status = mu_header_set_value (hdr, hdr_c, val_c, repl);
  free (hdr_c);
  free (val_c);
  
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot set header \"~A: ~A\" in message ~A",
		  scm_list_3 (header, value, mesg));
  
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_size, "mu-message-get-size", 1, 0, 0,
		   (SCM mesg),
		   "Returns size of the message @var{mesg}\n.")
#define FUNC_NAME s_scm_mu_message_get_size
{
  mu_message_t msg;
  size_t size;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  mu_message_size (msg, &size);
  return scm_from_size_t (size);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_lines, "mu-message-get-lines", 1, 0, 0,
		   (SCM mesg),
		   "Returns number of lines in the message @var{msg}.\n")
#define FUNC_NAME s_scm_mu_message_get_lines
{
  mu_message_t msg;
  size_t lines;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_lines (msg, &lines);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get number of lines in message ~A",
		  scm_list_1 (mesg));

  return scm_from_size_t (lines);
}
#undef FUNC_NAME

static SCM
filltime (struct tm *bd_time, int zoff, const char *zname)
{
  SCM result = scm_c_make_vector (11, SCM_UNDEFINED);

  SCM_SIMPLE_VECTOR_SET (result,0, scm_from_int (bd_time->tm_sec));
  SCM_SIMPLE_VECTOR_SET (result,1, scm_from_int (bd_time->tm_min));
  SCM_SIMPLE_VECTOR_SET (result,2, scm_from_int (bd_time->tm_hour));
  SCM_SIMPLE_VECTOR_SET (result,3, scm_from_int (bd_time->tm_mday));
  SCM_SIMPLE_VECTOR_SET (result,4, scm_from_int (bd_time->tm_mon));
  SCM_SIMPLE_VECTOR_SET (result,5, scm_from_int (bd_time->tm_year));
  SCM_SIMPLE_VECTOR_SET (result,6, scm_from_int (bd_time->tm_wday));
  SCM_SIMPLE_VECTOR_SET (result,7, scm_from_int (bd_time->tm_yday));
  SCM_SIMPLE_VECTOR_SET (result,8, scm_from_int (bd_time->tm_isdst));
  SCM_SIMPLE_VECTOR_SET (result,9, scm_from_int (zoff));
  SCM_SIMPLE_VECTOR_SET (result,10, (zname
				     ? scm_from_locale_string (zname)
				     : SCM_BOOL_F));
  return result;
}

SCM_DEFINE_PUBLIC (scm_mu_message_get_envelope, "mu-message-get-envelope", 1, 0, 0,
		   (SCM mesg),
		   "Returns envelope date of the message @var{mesg}.\n")
#define FUNC_NAME s_scm_mu_message_get_envelope
{
  mu_message_t msg;
  mu_envelope_t env = NULL;
  int status;
  const char *sender;
  const char *date;
  size_t dlen;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_get_envelope (msg, &env);
  if (status)
    mu_scm_error (FUNC_NAME, status, "cannot get envelope",
		  scm_list_1 (mesg));
  status = mu_envelope_sget_sender (env, &sender);
  if (status)
    mu_scm_error (FUNC_NAME, status, "cannot get envelope sender",
		  scm_list_1 (mesg));
  status = mu_envelope_sget_date (env, &date);
  if (status)
    mu_scm_error (FUNC_NAME, status, "cannot get envelope date",
		  scm_list_1 (mesg));
  dlen = strlen (date);
  if (date[dlen-1] == '\n')
    dlen--;
  return scm_string_append (scm_list_3 (scm_from_locale_string (sender),
					scm_from_locale_string (" "),
					scm_from_locale_stringn (date, dlen)));
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_envelope_date, "mu-message-get-envelope-date", 1, 0, 0,
		   (SCM mesg),
		   "Returns envelope date of the message @var{mesg}.\n")
#define FUNC_NAME s_scm_mu_message_get_envelope_date
{
  mu_message_t msg;
  mu_envelope_t env = NULL;
  int status;
  const char *sdate;
  struct tm tm;
  mu_timezone tz;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_get_envelope (msg, &env);
  if (status)
    mu_scm_error (FUNC_NAME, status, "cannot get envelope",
		  scm_list_1 (mesg));
  status = mu_envelope_sget_date (env, &sdate);
  if (status)
    mu_scm_error (FUNC_NAME, status, "cannot get envelope date",
		  scm_list_1 (mesg));
  status = mu_parse_ctime_date_time (&sdate, &tm, &tz);
  if (status)
    mu_scm_error (FUNC_NAME, status, "invalid envelope date",
		  scm_list_1 (scm_from_locale_string (sdate)));
  return filltime (&tm, tz.utc_offset, tz.tz_name);  
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_sender, "mu-message-get-sender", 1, 0, 0,
		   (SCM mesg),
	   "Returns email address of the sender of the message @var{mesg}.\n")
#define FUNC_NAME s_scm_mu_message_get_sender
{
  mu_message_t msg;
  mu_envelope_t env = NULL;
  int status;
  SCM ret;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_get_envelope (msg, &env);
  if (status == 0)
    {
      char *p = _get_envelope_sender (env);
      ret = scm_from_locale_string (p);
      free (p);
    }
  else
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get envelope of message ~A",
		  scm_list_1 (mesg));
  return ret;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_header, "mu-message-get-header", 2, 0, 0,
		   (SCM mesg, SCM header),
"Returns value of the header @var{header} from the message @var{mesg}.\n")
#define FUNC_NAME s_scm_mu_message_get_header
{
  mu_message_t msg;
  mu_header_t hdr;
  char *value = NULL;
  char *header_string;
  SCM ret;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_string (header), header, SCM_ARG2, FUNC_NAME);
  status = mu_message_get_header (msg, &hdr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message headers", SCM_BOOL_F);

  header_string = scm_to_locale_string (header);
  status = mu_header_aget_value (hdr, header_string, &value);
  free (header_string);
  switch (status)
    {
    case 0:
      ret = scm_from_locale_string (value);
      free (value);
      break;
      
    case MU_ERR_NOENT:
      ret = SCM_BOOL_F;
      break;

    default:
      mu_scm_error (FUNC_NAME, status,
		    "Cannot get header ~A from message ~A",
		    scm_list_2 (header, mesg));
    }

  return ret;
}
#undef FUNC_NAME

static int
string_sloppy_member (SCM lst, char *name)
{
  for(; SCM_CONSP (lst); lst = SCM_CDR(lst))
    {
      SCM car = SCM_CAR (lst);
      if (scm_is_string (car)
	  && mu_c_strncasecmp (scm_i_string_chars (car), name,
			       scm_i_string_length (car)) == 0)
	return 1;
    }
  return 0;
}

SCM_DEFINE_PUBLIC (scm_mu_message_get_header_fields, "mu-message-get-header-fields", 1, 1, 0,
		   (SCM mesg, SCM headers),
"Returns list of headers in the message @var{mesg}. optional argument\n" 
"@var{headers} gives a list of header names to restrict return value to.\n")
#define FUNC_NAME s_scm_mu_message_get_header_fields
{
  size_t i, nfields = 0;
  mu_message_t msg;
  mu_header_t hdr = NULL;
  SCM scm_first = SCM_EOL, scm_last = SCM_EOL;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  if (SCM_UNBNDP (headers))
    headers = SCM_EOL;
  else
    SCM_ASSERT (scm_is_pair (headers), headers, SCM_ARG2, FUNC_NAME);

  status = mu_message_get_header (msg, &hdr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message headers", SCM_BOOL_F);
  status = mu_header_get_field_count (hdr, &nfields);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get header field count", SCM_BOOL_F);
  
  for (i = 1; i <= nfields; i++)
    {
      SCM scm_name, scm_value, scm_new;
      char *name, *value;
      
      status = mu_header_aget_field_name (hdr, i, &name);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot get header field ~A, message ~A",
		      scm_list_2 (scm_from_size_t (i), mesg));
      
      if (!scm_is_null (headers) && string_sloppy_member (headers, name) == 0)
	continue;
      status = mu_header_aget_field_value (hdr, i, &value);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot get header value ~A, message ~A",
		      scm_list_2 (scm_from_size_t (i), mesg));

      scm_name = scm_from_locale_string (name);
      scm_value = scm_from_locale_string (value);

      scm_new = scm_cons (scm_cons (scm_name, scm_value), SCM_EOL);
      
      if (scm_is_null (scm_first))
	scm_first = scm_last = scm_new;
      else
	{
	  SCM_SETCDR (scm_last, scm_new);
	  scm_last = scm_new;
	}
    }
  return scm_first;
}
#undef FUNC_NAME
  
SCM_DEFINE_PUBLIC (scm_mu_message_set_header_fields, "mu-message-set-header-fields", 2, 1, 0,
		   (SCM mesg, SCM list, SCM replace),
"Set headers in the message @var{mesg} to those listed in @var{list},\n"
"which is a list of conses @code{(cons @var{header} @var{value})}.\n\n"
"Optional parameter @var{replace} specifies whether new header\n"
"values should replace the headers already present in the\n"
"message.")
#define FUNC_NAME s_scm_mu_message_set_header_fields
{
  mu_message_t msg;
  mu_header_t hdr;
  int repl = 0;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_null (list) || scm_is_pair (list),
	      list, SCM_ARG2, FUNC_NAME);
  if (!SCM_UNBNDP (replace))
    {
      SCM_ASSERT (scm_is_bool (replace), replace, SCM_ARG3, FUNC_NAME);
      repl = replace == SCM_BOOL_T;
    }

  status = mu_message_get_header (msg, &hdr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message headers", SCM_BOOL_F);

  for (; !scm_is_null (list); list = SCM_CDR (list))
    {
      SCM cell = SCM_CAR (list);
      SCM car, cdr;
      char *hdr_c, *val_c;
      
      SCM_ASSERT (scm_is_pair (cell), cell, SCM_ARGn, FUNC_NAME);
      car = SCM_CAR (cell);
      cdr = SCM_CDR (cell);
      SCM_ASSERT (scm_is_string (car), car, SCM_ARGn, FUNC_NAME);
      SCM_ASSERT (scm_is_string (cdr), cdr, SCM_ARGn, FUNC_NAME);
      hdr_c = scm_to_locale_string (car);
      val_c = scm_to_locale_string (cdr);
      status = mu_header_set_value (hdr, hdr_c, val_c, repl);
      free (hdr_c);
      free (val_c);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot set header value: message ~A, header ~A, value ~A",
		      scm_list_3 (mesg, car, cdr));

    }
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_delete, "mu-message-delete", 1, 1, 0,
		   (SCM mesg, SCM flag),
"Mark message @var{mesg} as deleted. Optional argument @var{flag} allows to\n"
"toggle the deletion mark. The message is deleted if it is @code{#t} and\n"
"undeleted if it is @code{#f}.")
#define FUNC_NAME s_scm_mu_message_delete
{
  mu_message_t msg;
  mu_attribute_t attr;
  int delete = 1;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  if (!SCM_UNBNDP (flag))
    {
      SCM_ASSERT (scm_is_bool (flag), flag, SCM_ARG2, FUNC_NAME);
      delete = flag == SCM_BOOL_T;
    }
  status = mu_message_get_attribute (msg, &attr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message attribute", SCM_BOOL_F);

  if (delete)
    status = mu_attribute_set_deleted (attr);
  else
    status = mu_attribute_unset_deleted (attr);
  
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Error setting message attribute", SCM_BOOL_F);

  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_flag, "mu-message-get-flag", 2, 0, 0,
		   (SCM mesg, SCM flag),
"Return the value of the attribute @var{flag} of the message @var{mesg}.")
#define FUNC_NAME s_scm_mu_message_get_flag
{
  mu_message_t msg;
  mu_attribute_t attr;
  int ret = 0;
  int status;
    
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_integer (flag), flag, SCM_ARG2, FUNC_NAME);

  status = mu_message_get_attribute (msg, &attr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message attribute", SCM_BOOL_F);
  
  switch (scm_to_int (flag))
    {
    case MU_ATTRIBUTE_ANSWERED:
      ret = mu_attribute_is_answered (attr);
      break;
      
    case MU_ATTRIBUTE_FLAGGED:
      ret = mu_attribute_is_flagged (attr);
      break;
      
    case MU_ATTRIBUTE_DELETED:
      ret = mu_attribute_is_deleted (attr);
      break;
      
    case MU_ATTRIBUTE_DRAFT:
      ret = mu_attribute_is_draft (attr);
      break;
      
    case MU_ATTRIBUTE_SEEN:
      ret = mu_attribute_is_seen (attr);
      break;
      
    case MU_ATTRIBUTE_READ:
      ret = mu_attribute_is_read (attr);
      break;
      
    case MU_ATTRIBUTE_MODIFIED:
      ret = mu_attribute_is_modified (attr);
      break;
      
    case MU_ATTRIBUTE_RECENT:
      ret = mu_attribute_is_recent (attr);
      break;
      
    default:
      mu_attribute_get_flags (attr, &ret);
      ret &= scm_to_int (flag);
    }
  return ret ? SCM_BOOL_T : SCM_BOOL_F;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_set_flag, "mu-message-set-flag", 2, 1, 0,
		   (SCM mesg, SCM flag, SCM value),
"Set the attribute @var{flag} in message @var{mesg}. If optional @var{value}\n"
"is @samp{#f}, the attribute is unset.\n")
#define FUNC_NAME s_scm_mu_message_set_flag
{
  mu_message_t msg;
  mu_attribute_t attr;
  int val = 1;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_integer (flag), flag, SCM_ARG2, FUNC_NAME);

  if (!SCM_UNBNDP (value))
    {
      SCM_ASSERT (scm_is_bool (value), value, SCM_ARG3, FUNC_NAME);
      val = value == SCM_BOOL_T;
    }
  
  status = mu_message_get_attribute (msg, &attr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message attribute", SCM_BOOL_F);

  status = 0;
  switch (scm_to_int (flag))
    {
    case MU_ATTRIBUTE_ANSWERED:
      if (val)
	status = mu_attribute_set_answered (attr);
      else
	status = mu_attribute_unset_answered (attr);
      break;
      
    case MU_ATTRIBUTE_FLAGGED:
      if (val)
	status = mu_attribute_set_flagged (attr);
      else
	status = mu_attribute_unset_flagged (attr);
      break;
      
    case MU_ATTRIBUTE_DELETED:
      if (val)
	status = mu_attribute_set_deleted (attr);
      else
	status = mu_attribute_unset_deleted (attr);
      break;
      
    case MU_ATTRIBUTE_DRAFT:
      if (val)
	status = mu_attribute_set_draft (attr);
      else
	status = mu_attribute_unset_draft (attr);
      break;
      
    case MU_ATTRIBUTE_SEEN:
      if (val)
	status = mu_attribute_set_seen (attr);
      else
	status = mu_attribute_unset_seen (attr);
      break;
      
    case MU_ATTRIBUTE_READ:
      if (val)
	status = mu_attribute_set_read (attr);
      else
	status = mu_attribute_unset_read (attr);
      break;
      
    case MU_ATTRIBUTE_MODIFIED:
      if (val)
	status = mu_attribute_set_modified (attr);
      else
	status = mu_attribute_clear_modified (attr);
      break;
      
    case MU_ATTRIBUTE_RECENT:
      if (val)
	status = mu_attribute_set_recent (attr);
      else
	status = mu_attribute_unset_recent (attr);
      break;
      
    default:
      if (val)
	status = mu_attribute_set_flags (attr, scm_to_int (flag));
    }
  
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Error setting message attribute", SCM_BOOL_F);
  
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_user_flag, "mu-message-get-user-flag", 2, 0, 0,
		   (SCM mesg, SCM flag),
"Return value of the user-defined attribute @var{flag} from the message @var{mesg}.")
#define FUNC_NAME s_scm_mu_message_get_user_flag
{
  mu_message_t msg;
  mu_attribute_t attr;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_integer (flag), flag, SCM_ARG2, FUNC_NAME);
  status = mu_message_get_attribute (msg, &attr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message attribute", SCM_BOOL_F);
  return mu_attribute_is_userflag (attr, scm_to_int (flag)) ?
                                   SCM_BOOL_T : SCM_BOOL_F;
}
#undef FUNC_NAME
  

SCM_DEFINE_PUBLIC (scm_mu_message_set_user_flag, "mu-message-set-user-flag", 2, 1, 0,
		   (SCM mesg, SCM flag, SCM value),
"Set user-defined attribute @var{flag} in the message @var{mesg}.\n"
"If optional argumen @var{value} is @samp{#f}, the attribute is unset.")
#define FUNC_NAME s_scm_mu_message_set_user_flag
{
  mu_message_t msg;
  mu_attribute_t attr;
  int set = 1;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  SCM_ASSERT (scm_is_integer (flag), flag, SCM_ARG2, FUNC_NAME);

  if (!SCM_UNBNDP (value))
    {
      SCM_ASSERT (scm_is_bool (value), value, SCM_ARG3, FUNC_NAME);
      set = value == SCM_BOOL_T;
    }
  
  status = mu_message_get_attribute (msg, &attr);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message attribute", SCM_BOOL_F);
  
  if (set)
    mu_attribute_set_userflag (attr, scm_to_int (flag));
  else
    mu_attribute_unset_userflag (attr, scm_to_int (flag));
  return SCM_UNSPECIFIED;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_port, "mu-message-get-port", 2, 1, 0,
		   (SCM mesg, SCM mode, SCM full),
"Returns a port associated with the message @var{mesg}. The @var{mode} is a\n"
"string defining operation mode of the stream. It may contain any of the\n"
"two characters: @samp{r} for reading, @samp{w} for writing.\n"
"If optional argument @var{full} is specified, it should be a boolean value.\n"
"If it is @samp{#t} then the returned port will allow access to any\n"
"part of the message (including headers). If it is @code{#f} then the port\n"
"accesses only the message body (the default).\n")
#define FUNC_NAME s_scm_mu_message_get_port
{
  mu_message_t msg;
  mu_stream_t stream = NULL;
  int status;
  char *str;
  SCM ret;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  SCM_ASSERT (scm_is_string (mode), mode, SCM_ARG2, FUNC_NAME);

  msg = mu_scm_message_get (mesg);

  if (!SCM_UNBNDP (full))
    {
      SCM_ASSERT (scm_is_bool (full), full, SCM_ARG3, FUNC_NAME);
      if (full == SCM_BOOL_T)
	{
	  status = mu_message_get_stream (msg, &stream);
	  if (status)
	    mu_scm_error (FUNC_NAME, status, "Cannot get message stream",
			  SCM_BOOL_F);
	}
    }

  if (!stream)
    {
      mu_body_t body = NULL;

      status = mu_message_get_body (msg, &body);
      if (status)
	mu_scm_error (FUNC_NAME, status, "Cannot get message body",
		      SCM_BOOL_F);
      status = mu_body_get_stream (body, &stream);
      if (status)
	mu_scm_error (FUNC_NAME, status, "Cannot get message body stream",
		      SCM_BOOL_F);
    }

  str = scm_to_locale_string (mode);
  ret = mu_port_make_from_stream (mesg, stream, scm_mode_bits (str));
  free (str);
  return ret;
}
#undef FUNC_NAME
  

SCM_DEFINE_PUBLIC (scm_mu_message_get_body, "mu-message-get-body", 1, 0, 0,
		   (SCM mesg),
		   "Returns message body for the message @var{mesg}.")
#define FUNC_NAME s_scm_mu_message_get_body
{
  mu_message_t msg;
  mu_body_t body = NULL;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_get_body (msg, &body);
  if (status)
    mu_scm_error (FUNC_NAME, status, "Cannot get message body", SCM_BOOL_F);
  return mu_scm_body_create (mesg, body);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_multipart_p, "mu-message-multipart?", 1, 0, 0,
		   (SCM mesg),
"Returns @code{#t} if @var{mesg} is a multipart @acronym{MIME} message.")
#define FUNC_NAME s_scm_mu_message_multipart_p
{
  mu_message_t msg;
  int ismime = 0;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  mu_message_is_multipart (msg, &ismime);
  return ismime ? SCM_BOOL_T : SCM_BOOL_F;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_num_parts, "mu-message-get-num-parts", 1, 0, 0,
		   (SCM mesg),
"Returns number of parts in a multipart @acronym{MIME} message @var{mesg}.\n"
"Returns @code{#f} if the argument is not a multipart message.")
#define FUNC_NAME s_scm_mu_message_get_num_parts
{
  mu_message_t msg;
  int ismime = 0;
  size_t nparts = 0;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  mu_message_is_multipart (msg, &ismime);
  if (!ismime)
    return SCM_BOOL_F;

  status = mu_message_get_num_parts (msg, &nparts);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get number of parts in the message ~A",
		  scm_list_1 (mesg));
  return scm_from_size_t (nparts);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_part, "mu-message-get-part", 2, 0, 0,
		   (SCM mesg, SCM part),
"Returns part #@var{part} of a multipart @acronym{MIME} message @var{mesg}.")
#define FUNC_NAME s_scm_mu_message_get_part
{
  mu_message_t msg, submsg;
  int ismime = 0;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  SCM_ASSERT (scm_is_integer (part), part, SCM_ARG2, FUNC_NAME);

  msg = mu_scm_message_get (mesg);
  mu_message_is_multipart (msg, &ismime);
  if (!ismime)
    return SCM_BOOL_F;

  status = mu_message_get_part (msg, scm_to_size_t (part), &submsg);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get number of part ~A from the message ~A",
		  scm_list_2 (part, mesg));

  return mu_scm_message_create (mesg, submsg);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_send, "mu-message-send", 1, 3, 0,
		   (SCM mesg, SCM mailer, SCM from, SCM to),
"Sends message @var{mesg}. Optional @var{mailer} overrides default mailer\n"
"settings. Optional @var{from} and @var{to} give sender and recever\n"
"addresses, respectively.\n")
#define FUNC_NAME s_scm_mu_message_send
{
  char *mailer_name;
  mu_address_t from_addr = NULL;
  mu_address_t to_addr = NULL;
  mu_mailer_t mailer_c = NULL;
  mu_message_t msg;
  int status;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  
  if (!SCM_UNBNDP (mailer) && mailer != SCM_BOOL_F)
    {
      SCM_ASSERT (scm_is_string (mailer), mailer, SCM_ARG2, FUNC_NAME);
      mailer_name = scm_to_locale_string (mailer);
    }
  else
    {
      SCM val = MU_SCM_SYMBOL_VALUE ("mu-mailer");
      mailer_name = scm_to_locale_string (val);
    }
  
  if (!SCM_UNBNDP (from) && from != SCM_BOOL_F)
    {
      char *s;
      int rc;
      
      SCM_ASSERT (scm_is_string (from), from, SCM_ARG3, FUNC_NAME);
      s = scm_to_locale_string (from);
      rc = mu_address_create (&from_addr, s);
      free (s);
      if (rc)
	mu_scm_error (FUNC_NAME, rc, "cannot create address",
		      scm_list_1 (from));
    }
  
  if (!SCM_UNBNDP (to) && to != SCM_BOOL_F)
    {
      char *s;
      int rc;
      
      SCM_ASSERT (scm_is_string (to), to, SCM_ARG4, FUNC_NAME);
      s = scm_to_locale_string (to);
      rc = mu_address_create (&to_addr, s);
      free (s);
      if (rc)
	mu_scm_error (FUNC_NAME, rc, "cannot create address",
		      scm_list_1 (to));
    }

  status = mu_mailer_create (&mailer_c, mailer_name);
  free (mailer_name);
  if (status)
    mu_scm_error (FUNC_NAME, status, "Cannot get create mailer", SCM_BOOL_F);

  if (scm_to_int (MU_SCM_SYMBOL_VALUE ("mu-debug")))
    {
      mu_debug_t debug = NULL;
      mu_mailer_get_debug (mailer_c, &debug);
      mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
    }

  status = mu_mailer_open (mailer_c, MU_STREAM_RDWR);
  if (status == 0)
    {
      status = mu_mailer_send_message (mailer_c, msg, from_addr, to_addr);
      if (status)
	mu_scm_error (FUNC_NAME, status, "Cannot send message", SCM_BOOL_F);

      mu_mailer_close (mailer_c);
    }
  else
    mu_scm_error (FUNC_NAME, status, "Cannot open mailer", SCM_BOOL_F);
  mu_mailer_destroy (&mailer_c);
  
  return status == 0 ? SCM_BOOL_T : SCM_BOOL_F;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_message_get_uid, "mu-message-get-uid", 1, 0, 0,
		   (SCM mesg),
	    "Returns UID of the message @var{mesg}\n")
#define FUNC_NAME s_scm_mu_message_get_uid
{
  mu_message_t msg;
  int status;
  size_t uid;
  
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG1, FUNC_NAME);
  msg = mu_scm_message_get (mesg);
  status = mu_message_get_uid (msg, &uid);
  if (status)
    mu_scm_error (FUNC_NAME, status, "Cannot get message uid", SCM_BOOL_F);
  return scm_from_size_t (uid);
}
#undef FUNC_NAME

/* Initialize the module */

void
mu_scm_message_init ()
{
  message_tag = scm_make_smob_type ("message", sizeof (struct mu_message));
  scm_set_smob_mark (message_tag, mu_scm_message_mark);
  scm_set_smob_free (message_tag, mu_scm_message_free);
  scm_set_smob_print (message_tag, mu_scm_message_print);

#include "mu_message.x"

}
