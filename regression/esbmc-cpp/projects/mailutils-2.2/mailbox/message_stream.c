/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

/* This file implements an MH draftfile stream: a read-only stream used
   to transparently pass MH draftfiles to mailers. The only difference
   between the usual RFC822 and MH draft is that the latter allows to use
   a string of dashes to separate the headers from the body. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include <mailutils/types.h>
#include <mailutils/address.h>
#include <mailutils/envelope.h>
#include <mailutils/message.h>
#include <mailutils/header.h>
#include <mailutils/body.h>
#include <mailutils/stream.h>
#include <mailutils/mutil.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/cstr.h>

struct _mu_rfc822_stream
{
  mu_stream_t stream;  /* Actual stream */
  char *envelope;
  size_t envelope_length; 
  size_t mark_offset;  /* Offset of the header separator */
  size_t mark_length;  /* Length of the header separator (not counting the
			  newline) */
};

static int
_mu_rfc822_read (mu_stream_t stream, char *optr, size_t osize,
		 mu_off_t offset, size_t *nbytes)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);

  offset += s->envelope_length;
  if (offset < s->mark_offset)
    {
      if (offset + osize >= s->mark_offset)
	osize = s->mark_offset - offset;
    }
  else
    offset += s->mark_length;
  return mu_stream_read (s->stream, optr, osize, offset, nbytes);
}
  
static int
_mu_rfc822_readline (mu_stream_t stream, char *optr, size_t osize,
		     mu_off_t offset, size_t *nbytes)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);
    
  offset += s->envelope_length;
  if (offset < s->mark_offset)
    {
      if (offset + osize >= s->mark_offset)
	{
	  int rc;
	  size_t n;
	  size_t rdsize = s->mark_offset - offset + 1;

	  rc = mu_stream_readline (s->stream, optr, rdsize, offset, &n);
	  if (rc == 0)
	    {
	      if (nbytes)
		*nbytes = n;
	    }
	  return rc;
	}
    }
  else
    offset += s->mark_length;

  return mu_stream_readline (s->stream, optr, osize, offset, nbytes);
}
  
static int
_mu_rfc822_size (mu_stream_t stream, mu_off_t *psize)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);
  int rc = mu_stream_size (s->stream, psize);
  
  if (rc == 0)
    *psize -= s->envelope_length + s->mark_length;
  return rc;
}
  
static int
_mu_rfc822_open (mu_stream_t stream)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);
  size_t offset, len;
  char *buffer = NULL;
  size_t bufsize = 0;
  int rc;

  offset = 0;
  while ((rc = mu_stream_getline (s->stream, &buffer, &bufsize,
				  offset, &len)) == 0
	 && len > 0)
    {
      if (offset == 0 && memcmp (buffer, "From ", 5) == 0)
	{
	  s->envelope_length = len;
	  s->envelope = strdup (buffer);
	  if (!s->envelope)
	    return ENOMEM;
	  s->envelope[len - 1] = 0;
	}
      else if (mu_mh_delim (buffer))
	{
	  s->mark_offset = offset;
	  s->mark_length = len - 1; /* do not count the terminating newline */
	  break;
	}

      offset += len;
    }
  free (buffer);
  return 0;
}

static int
_mu_rfc822_close (mu_stream_t stream)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);
  return mu_stream_close (s->stream);
}

static void
_mu_rfc822_destroy (mu_stream_t stream)
{
  struct _mu_rfc822_stream *s = mu_stream_get_owner (stream);

  free (s->envelope);
  if (s->stream)
    mu_stream_destroy (&s->stream, mu_stream_get_owner (s->stream));
  free (s);
}
    
int
mu_rfc822_stream_create (mu_stream_t *stream, mu_stream_t src, int flags)
{
  struct _mu_rfc822_stream *s;
  int rc;

  if (!flags)
    flags = MU_STREAM_READ;
  if (flags != MU_STREAM_READ)
    return EINVAL;

  s = calloc (1, sizeof (*s));
  if (s == NULL)
    return ENOMEM;

  s->stream = src;
  
  rc = mu_stream_create (stream, flags|MU_STREAM_NO_CHECK, s);
  if (rc)
    {
      free (s);
      return rc;
    }
  
  mu_stream_set_open (*stream, _mu_rfc822_open, s);
  mu_stream_set_close (*stream, _mu_rfc822_close, s);
  mu_stream_set_destroy (*stream, _mu_rfc822_destroy, s);
  mu_stream_set_readline (*stream, _mu_rfc822_readline, s);
  mu_stream_set_read (*stream, _mu_rfc822_read, s);
  mu_stream_set_size (*stream, _mu_rfc822_size, s);

  return 0;  
}



/* *************************** MH draft message **************************** */

static char *
skipws (char *p, size_t off)
{
  int len;
  for (p += off; *p && isspace (*p); p++)
    ;
  len = strlen (p);
  if (len > 0 && p[len-1] == '\n')
    p[len-1] = 0;
  return p;
}

struct _mu_rfc822_message
{
  char *from;
  char *date;
  mu_off_t body_start;
  mu_off_t body_end;
};

static int
restore_envelope (mu_stream_t str, struct _mu_rfc822_message **pmenv)
{
  size_t offset = 0;
  char *from = NULL;
  char *env_from = NULL;
  char *env_date = NULL;
  int rc;
  char *buffer = NULL;
  size_t bufsize = 0;
  size_t len;
  mu_off_t body_start, body_end;
  struct _mu_rfc822_stream *s822 = mu_stream_get_owner (str);

  if (s822->envelope)
    {
      char *s = s822->envelope + 5;
      char *p = strchr (s, ' ');
      size_t len;

      if (p)
	{
	  len = p - s;
	  env_from = malloc (len + 1);
	  if (!env_from)
	    return ENOMEM;
	  memcpy(env_from, s, len);
	  env_from[len] = 0;
	  env_date = strdup (p + 1);
	  if (!env_date)
	    {
	      free (env_from);
	      return ENOMEM;
	    }
	}
    }
  
  while ((rc = mu_stream_getline (str, &buffer, &bufsize, offset, &len)) == 0
	 && len > 0)
    {
      if (buffer[0] == '\n')
	break;
      offset += len;

      if (!env_from || !env_date)
	{
      	  if (!from && mu_c_strncasecmp (buffer, MU_HEADER_FROM,
				         sizeof (MU_HEADER_FROM) - 1) == 0)
	    from = strdup (skipws (buffer, sizeof (MU_HEADER_FROM)));
	  else if (!env_from
		   && mu_c_strncasecmp (buffer, MU_HEADER_ENV_SENDER,
				        sizeof (MU_HEADER_ENV_SENDER) - 1) == 0)
	    env_from = strdup (skipws (buffer, sizeof (MU_HEADER_ENV_SENDER)));
	  else if (!env_date
		   && mu_c_strncasecmp (buffer, MU_HEADER_ENV_DATE,
				        sizeof (MU_HEADER_ENV_DATE) - 1) == 0)
	    env_date = strdup (skipws (buffer, sizeof (MU_HEADER_ENV_DATE)));
	}
    }

  free (buffer);
  
  body_start = offset + 1;
  mu_stream_size (str, &body_end);
  
  if (!env_from)
    {
      if (from)
	{
	  mu_address_t addr;
	  
	  mu_address_create (&addr, from);
	  if (!addr
	      || mu_address_aget_email (addr, 1, &env_from))
	    env_from = strdup ("GNU-Mailutils");
	  mu_address_destroy (&addr);
	}
      else
	env_from = strdup ("GNU-MH");
    }
	  
  if (!env_date)
    {
      struct tm *tm;
      time_t t;
      char date[80]; /* FIXME: This size is way too big */

      time(&t);
      tm = gmtime(&t);
      mu_strftime (date, sizeof (date), "%a %b %e %H:%M:%S %Y", tm);
      env_date = strdup (date);
    }

  *pmenv = malloc (sizeof (**pmenv)
		   + strlen (env_from)
		   + strlen (env_date)
		   + 2);
  if (!*pmenv)
    {
      free (env_from);
      free (env_date);
      return ENOMEM;
    }
  
  (*pmenv)->from = (char*) (*pmenv + 1);
  (*pmenv)->date = (char*) ((*pmenv)->from + strlen (env_from) + 1);

  strcpy ((*pmenv)->from, env_from);
  strcpy ((*pmenv)->date, env_date);

  (*pmenv)->body_start = body_start;
  (*pmenv)->body_end = body_end;
  
  free (env_from);
  free (env_date);
  free (from);
  return 0;
}

static int
_env_msg_date (mu_envelope_t envelope, char *buf, size_t len, size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  struct _mu_rfc822_message *env = mu_message_get_owner (msg);
  
  if (!env || !env->date)
    return EINVAL;
  if (buf)
    {
      strncpy (buf, env->date, len);
      buf[len-1] = 0;
      if (pnwrite)
	*pnwrite = len;
    }
  else if (!pnwrite)
    return EINVAL;
  else
    *pnwrite = strlen (env->date);
  return 0;
}

static int
_env_msg_sender (mu_envelope_t envelope, char *buf, size_t len,
		 size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  struct _mu_rfc822_message *env = mu_message_get_owner (msg);
  
  if (!env || !env->from)
    return EINVAL;
  if (buf)
    {
      strncpy (buf, env->from, len);
      buf[len-1] = 0;
      if (pnwrite)
	*pnwrite = len;
    }
  else if (!pnwrite)
    return EINVAL;
  else
    *pnwrite = strlen (env->from);
    
  return 0;
}

static int
_body_size (mu_body_t body, size_t *size)
{
  mu_message_t msg = mu_body_get_owner (body);
  struct _mu_rfc822_message *mp = mu_message_get_owner (msg);

  if (size)
    *size = mp->body_end - mp->body_start;
  return 0;
}

static int 
_body_read (mu_stream_t stream, char *optr, size_t osize,
	    mu_off_t offset, size_t *nbytes)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  struct _mu_rfc822_message *mp = mu_message_get_owner (msg);
  mu_stream_t str;

  mu_message_get_stream (msg, &str);
  return mu_stream_read (str, optr, osize, mp->body_start + offset, nbytes);
}

static int
_body_readline (mu_stream_t stream, char *optr, size_t osize,
		mu_off_t offset, size_t *nbytes)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  struct _mu_rfc822_message *mp = mu_message_get_owner (msg);
  mu_stream_t str;

  mu_message_get_stream (msg, &str);
  return mu_stream_readline (str, optr, osize,
			     mp->body_start + offset, nbytes);
}

static int
_body_stream_size (mu_stream_t stream, mu_off_t *psize)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  struct _mu_rfc822_message *mp = mu_message_get_owner (msg);
  
  if (psize)
    *psize = mp->body_end - mp->body_start;
  return 0;
}

static int
_body_stream_flush (mu_stream_t str MU_ARG_UNUSED)
{
  return 0;
}

int
mu_stream_to_message (mu_stream_t instream, mu_message_t *pmsg)
{
  struct _mu_rfc822_message *mp;
  mu_envelope_t env;
  mu_message_t msg;
  mu_body_t body;
  mu_stream_t bstream;
  mu_stream_t draftstream;
  int rc;
  
  if ((rc = mu_rfc822_stream_create (&draftstream, instream, 0)))
    return rc;

  if ((rc = mu_stream_open (draftstream)))
    {
      mu_stream_destroy (&draftstream, mu_stream_get_owner (draftstream));
      return rc;
    }

  if ((rc = restore_envelope (draftstream, &mp)
       || (rc = mu_message_create (&msg, mp))))
    {
      mu_stream_destroy (&draftstream, mu_stream_get_owner (draftstream));
      return rc;
    }
  
  mu_message_set_stream (msg, draftstream, mp);
  
  if ((rc = mu_envelope_create (&env, msg)))
    {
      mu_stream_destroy (&draftstream, mu_stream_get_owner (draftstream));
      return rc;
    }
  
  mu_envelope_set_date (env, _env_msg_date, msg);
  mu_envelope_set_sender (env, _env_msg_sender, msg);
  mu_message_set_envelope (msg, env, mp);

  mu_body_create (&body, msg);
  mu_stream_create (&bstream,  MU_STREAM_RDWR | MU_STREAM_SEEKABLE, body);

  mu_stream_set_read (bstream, _body_read, body);
  mu_stream_set_readline (bstream, _body_readline, body);
  mu_stream_set_size (bstream, _body_stream_size, body);
  mu_stream_set_flush (bstream, _body_stream_flush, body);
  mu_body_set_stream (body, bstream, msg);
  mu_body_set_size (body, _body_size, msg);
  mu_message_set_body (msg, body, mp);

  *pmsg = msg;
  return 0;
}
