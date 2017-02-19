/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>

#include <sys/time.h>

#include <mailutils/cstr.h>
#include <mailutils/address.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/iterator.h>
#include <mailutils/list.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/header.h>
#include <mailutils/body.h>
#include <mailutils/mailbox.h>
#include <mailutils/message.h>
#include <mailutils/argcv.h>
#include <mailutils/mutil.h>
#include <mailutils/mime.h>
#include <mailutils/io.h>

#include <mailer0.h>

static char *mailer_url_default;

/* FIXME: I'd like to check that the URL is valid, but that requires that the
   mailers already be registered! */
int
mu_mailer_set_url_default (const char *url)
{
  char *n = NULL;

  if (!url)
    return EINVAL;

  if ((n = strdup (url)) == NULL)
    return ENOMEM;

  if (mailer_url_default)
    free (mailer_url_default);

  mailer_url_default = n;

  return 0;
}

int
mu_mailer_get_url_default (const char **url)
{
  if (!url)
    return EINVAL;

  if (mailer_url_default)
    *url = mailer_url_default;
  else
    *url = MAILER_URL_DEFAULT;

  return 0;
}

static void
set_default_debug (mu_mailer_t mailer)
{
  mu_log_level_t level = mu_global_debug_level ("mailer");
  if (level)
    {
      mu_debug_t debug;
      if (mu_mailer_get_debug (mailer, &debug))
	return;
      mu_debug_set_level (debug, level);
    }
}

int
mu_mailer_create_from_url (mu_mailer_t *pmailer, mu_url_t url)
{
  mu_record_t record;

  if (mu_registrar_lookup_url (url, MU_FOLDER_ATTRIBUTE_FILE, &record,
			       NULL) == 0)
    {
      int (*m_init) (mu_mailer_t) = NULL;

      mu_record_get_mailer (record, &m_init);
      if (m_init)
        {
	  int status;
	  mu_mailer_t mailer;
	  int (*u_init) (mu_url_t) = NULL;
	  
	  /* Allocate memory for mailer.  */
	  mailer = calloc (1, sizeof (*mailer));
	  if (mailer == NULL)
	    return ENOMEM;

	  status = mu_monitor_create (&mailer->monitor, 0, mailer);
	  if (status)
	    {
	      mu_mailer_destroy (&mailer);
	      return status;
	    }

	  status = m_init (mailer);
	  if (status)
	    {
	      mu_mailer_destroy (&mailer);
	      return status;
	    }

	  mu_record_get_url (record, &u_init);
	  if (u_init && (status = u_init (url)) != 0)
	    {
	      mu_mailer_destroy (&mailer);
	      return status;
	    }
	  
	  mailer->url = url;
	  *pmailer = mailer;

	  set_default_debug (mailer);
	  return status;
	}
    }
  
    return MU_ERR_MAILER_BAD_URL;
}

int
mu_mailer_create (mu_mailer_t * pmailer, const char *name)
{
  int status;
  mu_url_t url;

  if (name == NULL)
    mu_mailer_get_url_default (&name);

  status = mu_url_create (&url, name);
  if (status)
    return status;
  status = mu_url_parse (url);
  if (status == 0)
    status = mu_mailer_create_from_url (pmailer, url);
  if (status)
    mu_url_destroy (&url);
  return status;
}

void
mu_mailer_destroy (mu_mailer_t * pmailer)
{
  if (pmailer && *pmailer)
    {
      mu_mailer_t mailer = *pmailer;
      mu_monitor_t monitor = mailer->monitor;

      if (mailer->observable)
	{
	  mu_observable_notify (mailer->observable, MU_EVT_MAILER_DESTROY,
				mailer);
	  mu_observable_destroy (&mailer->observable, mailer);
	}

      /* Call the object destructor.  */
      if (mailer->_destroy)
	mailer->_destroy (mailer);

      mu_monitor_wrlock (monitor);

      if (mailer->stream)
	{
	  /* FIXME: Should be the client responsability to close this?  */
	  /* mu_stream_close (mailer->stream); */
	  mu_stream_destroy (&(mailer->stream), mailer);
	}

      if (mailer->url)
	mu_url_destroy (&(mailer->url));

      if (mailer->debug)
	mu_debug_destroy (&(mailer->debug), mailer);

      if (mailer->property)
	mu_property_destroy (&(mailer->property), mailer);

      free (mailer);
      *pmailer = NULL;
      mu_monitor_unlock (monitor);
      mu_monitor_destroy (&monitor, mailer);
    }
}


/* -------------- stub functions ------------------- */

int
mu_mailer_open (mu_mailer_t mailer, int flag)
{
  if (mailer == NULL || mailer->_open == NULL)
    return ENOSYS;
  return mailer->_open (mailer, flag);
}

int
mu_mailer_close (mu_mailer_t mailer)
{
  if (mailer == NULL || mailer->_close == NULL)
    return ENOSYS;
  return mailer->_close (mailer);
}


int
mu_mailer_check_from (mu_address_t from)
{
  size_t n = 0;

  if (!from)
    return EINVAL;

  if (mu_address_get_count (from, &n) || n != 1)
    return MU_ERR_MAILER_BAD_FROM;

  if (mu_address_get_email_count (from, &n) || n == 0)
    return MU_ERR_MAILER_BAD_FROM;

  return 0;
}

int
mu_mailer_check_to (mu_address_t to)
{
  size_t count = 0;
  size_t emails = 0;
  size_t groups = 0;

  if (!to)
    return EINVAL;

  if (mu_address_get_count (to, &count))
    return MU_ERR_MAILER_BAD_TO;

  if (mu_address_get_email_count (to, &emails))
    return MU_ERR_MAILER_BAD_TO;

  if (emails == 0)
    return MU_ERR_MAILER_NO_RCPT_TO;

  if (mu_address_get_group_count (to, &groups))
    return MU_ERR_MAILER_BAD_TO;

  if (count - emails - groups != 0)
    /* then not everything is a group or an email address */
    return MU_ERR_MAILER_BAD_TO;

  return 0;
}

static void
save_fcc (mu_message_t msg)
{
  mu_header_t hdr;
  size_t count = 0, i;
  char buf[512], *fcc;
  
  if (mu_message_get_header (msg, &hdr))
    return;

  if (mu_header_get_value (hdr, MU_HEADER_FCC, NULL, 0, NULL))
    return;
  
  mu_header_get_field_count (hdr, &count);
  for (i = 1; i <= count; i++)
    {
      mu_mailbox_t mbox;
      
      mu_header_get_field_name (hdr, i, buf, sizeof buf, NULL);
      if (mu_c_strcasecmp (buf, MU_HEADER_FCC) == 0
	  && mu_header_aget_field_value (hdr, i, &fcc) == 0)
	{
	  int i, argc;
	  char **argv;
	  
	  mu_argcv_get (fcc, ",", NULL, &argc, &argv);
	  for (i = 0; i < argc; i += 2)
	    {
	      if (mu_mailbox_create_default (&mbox, argv[i]))
		continue; /*FIXME: error message?? */
	      if (mu_mailbox_open (mbox,
				   MU_STREAM_RDWR | MU_STREAM_CREAT
				   | MU_STREAM_APPEND) == 0)
		{
		  mu_mailbox_append_message (mbox, msg);
		  mu_mailbox_flush (mbox, 0);
		}
	      mu_mailbox_close (mbox);
	      mu_mailbox_destroy (&mbox);
	    }
	  mu_argcv_free (argc, argv);
	  free (fcc);
	}
    }
}

static int
_set_from (mu_address_t *pfrom, mu_message_t msg, mu_address_t from,
	   mu_mailer_t mailer)
{
  int status = 0;
  char *mail_from;
  mu_header_t header = NULL;

  *pfrom = NULL;
  
  /* Get MAIL_FROM from FROM, the message, or the environment. */
  if (!from)
    {
      const char *type;
      
      if ((status = mu_message_get_header (msg, &header)) != 0)
	return status;
      
      status = mu_header_aget_value (header, MU_HEADER_FROM, &mail_from);
      
      switch (status)
	{
	default:
	  return status;

	  /* Use the From: header. */
	case 0:
	  MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE,
		     "mu_mailer_send_message(): using From: %s\n",
		     mail_from);
	    
	  status = mu_address_create (pfrom, mail_from);
	  free (mail_from);
	  break;

	case MU_ERR_NOENT:
	  if (mu_property_sget_value (mailer->property, "TYPE", &type) == 0
	      && strcmp (type, "SENDMAIL") == 0)
	    return 0;
	  
	  /* Use the environment. */
	  mail_from = mu_get_user_email (NULL);

	  if (mail_from)
            MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE,
		       "mu_mailer_send_message(): using user's address: %s\n",
		       mail_from);
	  else
            MU_DEBUG (mailer->debug, MU_DEBUG_ERROR,
		      "mu_mailer_send_message(): no user's address, failing\n");

	  if (!mail_from)
	    return errno;

	  status = mu_address_create (pfrom, mail_from);
	  /* FIXME: should we add the From: header? */
	  break;
	}
    }

  return status;
}

static int
create_part (mu_mime_t mime, mu_stream_t istr, 
	     size_t fragsize, size_t n, size_t nparts, char *msgid)
{
  int status = 0;
  mu_message_t newmsg;
  mu_header_t newhdr;
  mu_body_t body;
  mu_stream_t ostr;
  char buffer[512], *str;
  size_t slen;
  
  mu_message_create (&newmsg, NULL);
  mu_message_get_header (newmsg, &newhdr); 

  str = NULL;
  slen = 0;
  mu_asnprintf (&str, &slen,
		"message/partial; id=\"%s\"; number=%lu; total=%lu",
		msgid, (unsigned long)n, (unsigned long)nparts);
  mu_header_append (newhdr, MU_HEADER_CONTENT_TYPE, str);
  mu_asnprintf (&str, &slen, "part %lu of %lu",
		(unsigned long)n, (unsigned long)nparts);
  mu_header_append (newhdr, MU_HEADER_CONTENT_DESCRIPTION, str);
  free (str);
  
  mu_message_get_body (newmsg, &body);
  mu_body_get_stream (body, &ostr);

  mu_stream_seek (ostr, 0, SEEK_SET);

  while (fragsize)
    {
      size_t rds = fragsize;
      if (rds > sizeof buffer)
	rds = sizeof buffer;
      
      status = mu_stream_sequential_read (istr, buffer, rds, &rds);
      if (status || rds == 0)
	break;
      status = mu_stream_sequential_write (ostr, buffer, rds);
      if (status)
	break;
      fragsize -= rds;
    }
  if (status == 0)
    {
      mu_mime_add_part (mime, newmsg);
      mu_message_unref (newmsg);
    }
  return status;
}

static void
merge_headers (mu_message_t newmsg, mu_header_t hdr)
{
  size_t i, count;
  mu_header_t newhdr;
  
  mu_message_get_header (newmsg, &newhdr);
  mu_header_get_field_count (hdr, &count);
  for (i = 1; i <= count; i++)
    {
      const char *fn, *fv;

      mu_header_sget_field_name (hdr, i, &fn);
      mu_header_sget_field_value (hdr, i, &fv);
      if (mu_c_strcasecmp (fn, MU_HEADER_MESSAGE_ID) == 0)
	continue;
      else if (mu_c_strcasecmp (fn, MU_HEADER_MIME_VERSION) == 0)
	mu_header_append (newhdr, "X-Orig-" MU_HEADER_MIME_VERSION,
			  fv);
      else if (mu_c_strcasecmp (fn, MU_HEADER_CONTENT_TYPE) == 0)
	mu_header_append (newhdr, "X-Orig-" MU_HEADER_CONTENT_TYPE,
			  fv);
      else if (mu_c_strcasecmp (fn, MU_HEADER_CONTENT_DESCRIPTION) == 0)
	mu_header_append (newhdr, "X-Orig-" MU_HEADER_CONTENT_DESCRIPTION,
			  fv);
      else
	mu_header_append (newhdr, fn, fv);
    }
}
  

int
send_fragments (mu_mailer_t mailer,
		mu_header_t hdr,
		mu_stream_t str,
		size_t nparts, size_t fragsize,
		struct timeval *delay,
		mu_address_t from, mu_address_t to)
{
  int status;
  size_t i;
  char *msgid = NULL;
  
  if (mu_header_aget_value (hdr, MU_HEADER_MESSAGE_ID, &msgid))
    mu_rfc2822_msg_id (0, &msgid);
  
  for (i = 1; i <= nparts; i++)
    {
      mu_message_t newmsg;
      mu_mime_t mime;
		  
      mu_mime_create (&mime, NULL, 0);
      status = create_part (mime, str, fragsize, i, nparts, msgid);
      if (status)
	break;

      mu_mime_get_message (mime, &newmsg);
      merge_headers (newmsg, hdr);
      
      status = mailer->_send_message (mailer, newmsg, from, to);
      mu_mime_destroy (&mime);
      if (status)
	break;
      if (delay)
	{
	  struct timeval t = *delay;
	  select (0, NULL, NULL, NULL, &t);
	}
    }
  free (msgid);
  return status;
}

int
mu_mailer_send_fragments (mu_mailer_t mailer,
			  mu_message_t msg,
			  size_t fragsize, struct timeval *delay,
			  mu_address_t from, mu_address_t to)
{
  int status;
  mu_address_t sender_addr = NULL;
  
  if (mailer == NULL)
    return EINVAL;
  if (mailer->_send_message == NULL)
    return ENOSYS;

  status = _set_from (&sender_addr, msg, from, mailer);
  if (status)
    return status;
  if (sender_addr)
    from = sender_addr;
  
  if ((!from || (status = mu_mailer_check_from (from)) == 0)
      && (!to || (status = mu_mailer_check_to (to)) == 0))
    {
      save_fcc (msg);
      if (fragsize == 0)
	status = mailer->_send_message (mailer, msg, from, to);
      else
	{
	  mu_header_t hdr;
	  mu_body_t body;
	  size_t bsize;
	  size_t nparts;
	  
	  /* Estimate the number of messages to be sent. */
	  mu_message_get_header (msg, &hdr);

	  mu_message_get_body (msg, &body);
	  mu_body_size (body, &bsize);

	  nparts = bsize + fragsize - 1;
	  if (nparts < bsize) /* overflow */
	    return EINVAL;
	  nparts /= fragsize;

	  if (nparts == 1)
	    status = mailer->_send_message (mailer, msg, from, to);
	  else
	    {
	      mu_stream_t str;
	      mu_body_get_stream (body, &str);
	      
	      status = send_fragments (mailer, hdr, str, nparts, fragsize,
				       delay, from, to);
	    }
	}
    }
  mu_address_destroy (&sender_addr);
  return status;
}

int
mu_mailer_send_message (mu_mailer_t mailer, mu_message_t msg,
			mu_address_t from, mu_address_t to)
{
  return mu_mailer_send_fragments (mailer, msg, 0, NULL, from, to);
}

int
mu_mailer_set_stream (mu_mailer_t mailer, mu_stream_t stream)
{
  if (mailer == NULL)
    return EINVAL;
  mailer->stream = stream;
  return 0;
}

int
mu_mailer_get_stream (mu_mailer_t mailer, mu_stream_t * pstream)
{
  if (mailer == NULL)
    return EINVAL;
  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *pstream = mailer->stream;
  return 0;
}

int
mu_mailer_get_observable (mu_mailer_t mailer, mu_observable_t * pobservable)
{
  /* FIXME: I should check for invalid types */
  if (mailer == NULL)
    return EINVAL;
  if (pobservable == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (mailer->observable == NULL)
    {
      int status = mu_observable_create (&(mailer->observable), mailer);
      if (status != 0)
	return status;
    }
  *pobservable = mailer->observable;
  return 0;
}

int
mu_mailer_get_property (mu_mailer_t mailer, mu_property_t * pproperty)
{
  if (mailer == NULL)
    return EINVAL;
  if (pproperty == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (mailer->property == NULL)
    {
      int status = mu_property_create (&(mailer->property), mailer);
      if (status != 0)
	return status;
    }
  *pproperty = mailer->property;
  return 0;
}

int
mu_mailer_set_debug (mu_mailer_t mailer, mu_debug_t debug)
{
  if (mailer == NULL)
    return EINVAL;
  mu_debug_destroy (&(mailer->debug), mailer);
  mailer->debug = debug;
  return 0;
}

int
mu_mailer_get_debug (mu_mailer_t mailer, mu_debug_t * pdebug)
{
  if (mailer == NULL)
    return EINVAL;
  if (pdebug == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (mailer->debug == NULL)
    {
      int status = mu_debug_create (&(mailer->debug), mailer);
      if (status != 0)
	return status;
    }
  *pdebug = mailer->debug;
  return 0;
}

int
mu_mailer_get_url (mu_mailer_t mailer, mu_url_t * purl)
{
  if (!mailer)
    return EINVAL;
  if (!purl)
    return MU_ERR_OUT_PTR_NULL;
  *purl = mailer->url;
  return 0;
}
