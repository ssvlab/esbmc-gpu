/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2009,
   2010 Free Software Foundation, Inc.

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
# include <config.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pwd.h>

#include <message0.h>

#include <mailutils/cctype.h>
#include <mailutils/address.h>
#include <mailutils/attribute.h>
#include <mailutils/auth.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/envelope.h>
#include <mailutils/errno.h>
#include <mailutils/folder.h>
#include <mailutils/header.h>
#include <mailutils/mailbox.h>
#include <mailutils/mutil.h>
#include <mailutils/observer.h>
#include <mailutils/stream.h>
#include <mailutils/mu_auth.h>
#include <mailutils/nls.h>
#include <mailutils/md5.h>
#include <mailutils/io.h>

#define MESSAGE_MODIFIED 0x10000;

static int message_read   (mu_stream_t is, char *buf, size_t buflen,
			   mu_off_t off, size_t *pnread );
static int message_write  (mu_stream_t os, const char *buf, size_t buflen,
			   mu_off_t off, size_t *pnwrite);
static int message_get_transport2 (mu_stream_t stream, mu_transport_t *pin, 
                                   mu_transport_t *pout);
static int message_sender (mu_envelope_t envelope, char *buf, size_t len,
			   size_t *pnwrite);
static int message_date   (mu_envelope_t envelope, char *buf, size_t len,
			   size_t *pnwrite);
static int message_stream_size (mu_stream_t stream, mu_off_t *psize);
static int message_header_fill (mu_header_t header, char *buffer,
			        size_t buflen, mu_off_t off,
				size_t * pnread);
static int message_body_read (mu_stream_t stream,  char *buffer,
			      size_t n, mu_off_t off, size_t *pn);

/*  Allocate ressources for the mu_message_t.  */
int
mu_message_create (mu_message_t *pmsg, void *owner)
{
  mu_message_t msg;
  int status;

  if (pmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;
  msg = calloc (1, sizeof (*msg));
  if (msg == NULL)
    return ENOMEM;
  status = mu_monitor_create (&(msg->monitor), 0, msg);
  if (status != 0)
    {
      free (msg);
      return status;
    }
  msg->owner = owner;
  msg->ref = 1;
  *pmsg = msg;
  return 0;
}

void
mu_message_destroy (mu_message_t *pmsg, void *owner)
{
  if (pmsg && *pmsg)
    {
      mu_message_t msg = *pmsg;
      mu_monitor_t monitor = msg->monitor;
      int destroy_lock = 0;

      mu_monitor_wrlock (monitor);
      /* Note: msg->ref may be incremented by mu_message_ref without
	 additional checking for its owner, therefore decrementing
	 it must also occur independently of the owner checking. Due
	 to this inconsistency ref may reach negative values, which
	 is very unfortunate.

	 The `owner' stuff is a leftover from older mailutils versions.
	 There is an ongoing attempt to remove it in the stream-cleanup
	 branch. When it is ready, it will be merged to the HEAD and this
	 will finally resolve this issue. */
      if (msg->ref > 0)
	msg->ref--;
      if ((msg->owner && msg->owner == owner)
	  || (msg->owner == NULL && msg->ref <= 0))
	{
	  destroy_lock =  1;
	  /* Notify the listeners.  */
	  /* FIXME: to be removed since we do not support this event.  */
	  if (msg->observable)
	    {
	      mu_observable_notify (msg->observable, MU_EVT_MESSAGE_DESTROY,
				    msg);
	      mu_observable_destroy (&(msg->observable), msg);
	    }

	  /* Envelope.  */
	  if (msg->envelope)
	    mu_envelope_destroy (&(msg->envelope), msg);

	  /* Header.  */
	  if (msg->header)
	    mu_header_destroy (&(msg->header), msg);

	  /* Body.  */
	  if (msg->body)
	    mu_body_destroy (&(msg->body), msg);

	  /* Attribute.  */
	  if (msg->attribute)
	    mu_attribute_destroy (&(msg->attribute), msg);

	  /* Stream.  */
	  if (msg->stream)
	    mu_stream_destroy (&(msg->stream), msg);

	  /*  Mime.  */
	  if (msg->mime)
	    mu_mime_destroy (&(msg->mime));

	  /* Loose the owner.  */
	  msg->owner = NULL;

	  /* Mailbox maybe created floating i.e they were created
	     implicitely by the message when doing something like:
	     mu_message_create (&msg, "pop://localhost/msgno=2", NULL);
	     mu_message_create (&msg, "imap://localhost/alain;uid=xxxxx", NULL);
	     althought the semantics about this is still flaky we our
	     making some provisions here for it.
	     if (msg->floating_mailbox && msg->mailbox)
	     mu_mailbox_destroy (&(msg->mailbox));
	  */
	  
	  if (msg->ref <= 0)
	    free (msg);
	}
      mu_monitor_unlock (monitor);
      if (destroy_lock)
	mu_monitor_destroy (&monitor, msg);
      /* Loose the link */
      *pmsg = NULL;
    }
}

int
mu_message_create_copy (mu_message_t *to, mu_message_t from)
{
  int status = 0;
  mu_stream_t fromstr = NULL;
  mu_stream_t tostr = NULL;
  mu_off_t off = 0;
  size_t n = 0;
  char buf[512];

  if (!to)
    return MU_ERR_OUT_PTR_NULL;
  if (!from)
    return EINVAL;

  if((status = mu_message_create (to, NULL)))
    return status;

  mu_message_get_stream (from, &fromstr);
  mu_message_get_stream (*to, &tostr);

  while (
      (status = mu_stream_readline (fromstr, buf, sizeof(buf), off, &n)) == 0
	 &&
      n > 0
      )
    {
      mu_stream_write (tostr, buf, n, off, NULL);
      off += n;
    }

  if(status)
    mu_message_destroy(to, NULL);
  
  return status;
}

int
mu_message_ref (mu_message_t msg)
{
  if (msg)
    {
      mu_monitor_wrlock (msg->monitor);
      msg->ref++;
      mu_monitor_unlock (msg->monitor);
    }
  return 0;
}

void *
mu_message_get_owner (mu_message_t msg)
{
  return (msg == NULL) ? NULL : msg->owner;
}

int
mu_message_is_modified (mu_message_t msg)
{
  int mod = 0;
  if (msg)
    {
      mod |= mu_header_is_modified (msg->header);
      mod |= mu_attribute_is_modified (msg->attribute);
      mod |= mu_body_is_modified (msg->body);
      mod |= msg->flags;
    }
  return mod;
}

int
mu_message_clear_modified (mu_message_t msg)
{
  if (msg)
    {
      if (msg->header)
	mu_header_clear_modified (msg->header);
      if (msg->attribute)
	mu_attribute_clear_modified (msg->attribute);
      if (msg->body)
	mu_body_clear_modified (msg->body);
      msg->flags &= ~MESSAGE_MODIFIED;
    }
  return 0;
}

int
mu_message_get_mailbox (mu_message_t msg, mu_mailbox_t *pmailbox)
{
  if (msg == NULL)
    return EINVAL;
  if (pmailbox == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *pmailbox = msg->mailbox;
  return 0;
}

int
mu_message_set_mailbox (mu_message_t msg, mu_mailbox_t mailbox, void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->mailbox = mailbox;
  return 0;
}

int
mu_message_get_header (mu_message_t msg, mu_header_t *phdr)
{
  if (msg == NULL)
    return EINVAL;
  if (phdr == NULL)
    return MU_ERR_OUT_PTR_NULL;

  /* Is it a floating mesg */
  if (msg->header == NULL)
    {
      mu_header_t header;
      int status = mu_header_create (&header, NULL, 0, msg);
      if (status != 0)
	return status;
      if (msg->stream)
	{
	  /* Was it created by us?  */
	  mu_message_t mesg = mu_stream_get_owner (msg->stream);
	  if (mesg != msg)
	    mu_header_set_fill (header, message_header_fill, msg);
	}
      msg->header = header;
    }
  *phdr = msg->header;
  return 0;
}

int
mu_message_set_header (mu_message_t msg, mu_header_t hdr, void *owner)
{
  if (msg == NULL )
    return EINVAL;
  if (msg->owner != owner)
     return EACCES;
  /* Make sure we destroy the old if it was own by the mesg */
  /* FIXME:  I do not know if somebody has already a ref on this ? */
  if (msg->header)
    mu_header_destroy (&(msg->header), msg);
  msg->header = hdr;
  msg->flags |= MESSAGE_MODIFIED;
  return 0;
}

int
mu_message_get_body (mu_message_t msg, mu_body_t *pbody)
{
  if (msg == NULL)
    return EINVAL;
  if (pbody == NULL)
    return MU_ERR_OUT_PTR_NULL;

  /* Is it a floating mesg.  */
  if (msg->body == NULL)
    {
      mu_body_t body;
      int status = mu_body_create (&body, msg);
      if (status != 0)
	return status;
      /* If a stream is already set use it to create the body stream.  */
      if (msg->stream)
	{
	  /* Was it created by us?  */
	  mu_message_t mesg = mu_stream_get_owner (msg->stream);
	  if (mesg != msg)
	    {
	      mu_stream_t stream;
	      int flags = 0;
	      mu_stream_get_flags (msg->stream, &flags);
	      if ((status = mu_stream_create (&stream, flags, body)) != 0)
		{
		  mu_body_destroy (&body, msg);
		  return status;
		}
	      mu_stream_set_read (stream, message_body_read, body);
	      mu_stream_setbufsiz (stream, 128);
	      mu_body_set_stream (body, stream, msg);
	    }
	}
      msg->body = body;
    }
  *pbody = msg->body;
  return 0;
}

int
mu_message_set_body (mu_message_t msg, mu_body_t body, void *owner)
{
  if (msg == NULL )
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  /* Make sure we destoy the old if it was own by the mesg.  */
  /* FIXME:  I do not know if somebody has already a ref on this ? */
  if (msg->body)
    mu_body_destroy (&(msg->body), msg);
  msg->body = body;
  msg->flags |= MESSAGE_MODIFIED;
  return 0;
}

int
mu_message_set_stream (mu_message_t msg, mu_stream_t stream, void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  /* Make sure we destoy the old if it was own by the mesg.  */
  /* FIXME:  I do not know if somebody has already a ref on this ? */
  if (msg->stream)
    mu_stream_destroy (&(msg->stream), msg);
  msg->stream = stream;
  msg->flags |= MESSAGE_MODIFIED;
  return 0;
}

int
mu_message_get_stream (mu_message_t msg, mu_stream_t *pstream)
{
  if (msg == NULL)
    return EINVAL;
  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (msg->stream == NULL)
    {
      mu_stream_t stream;
      int status;
      status = mu_stream_create (&stream, MU_STREAM_RDWR, msg);
      if (status != 0)
	return status;
      mu_stream_set_read (stream, message_read, msg);
      mu_stream_set_write (stream, message_write, msg);
      mu_stream_set_get_transport2 (stream, message_get_transport2, msg);
      mu_stream_set_size (stream, message_stream_size, msg);
      mu_stream_set_flags (stream, MU_STREAM_RDWR);
      msg->stream = stream;
    }

  *pstream = msg->stream;
  return 0;
}

int
mu_message_set_lines (mu_message_t msg, int (*_lines)
		   (mu_message_t, size_t *), void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_lines = _lines;
  return 0;
}

int
mu_message_lines (mu_message_t msg, size_t *plines)
{
  size_t hlines, blines;
  int ret = 0;

  if (msg == NULL)
    return EINVAL;
  /* Overload.  */
  if (msg->_lines)
    return msg->_lines (msg, plines);
  if (plines)
    {
      hlines = blines = 0;
      if ( ( ret = mu_header_lines (msg->header, &hlines) ) == 0 )
	      ret = mu_body_lines (msg->body, &blines);
      *plines = hlines + blines;
    }
  return ret;
}

int
mu_message_set_size (mu_message_t msg, int (*_size)
		  (mu_message_t, size_t *), void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_size = _size;
  return 0;
}

int
mu_message_size (mu_message_t msg, size_t *psize)
{
  size_t hsize, bsize;
  int ret = 0;

  if (msg == NULL)
    return EINVAL;
  /* Overload ? */
  if (msg->_size)
    return msg->_size (msg, psize);
  if (psize)
    {
      mu_header_t hdr = NULL;
      mu_body_t body = NULL;
      
      hsize = bsize = 0;
      mu_message_get_header (msg, &hdr);
      mu_message_get_body (msg, &body);
      if ( ( ret = mu_header_size (hdr, &hsize) ) == 0 )
	ret = mu_body_size (body, &bsize);
      *psize = hsize + bsize;
    }
  return ret;
}

int
mu_message_get_envelope (mu_message_t msg, mu_envelope_t *penvelope)
{
  if (msg == NULL)
    return EINVAL;
  if (penvelope == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (msg->envelope == NULL)
    {
      mu_envelope_t envelope;
      int status = mu_envelope_create (&envelope, msg);
      if (status != 0)
	return status;
      mu_envelope_set_sender (envelope, message_sender, msg);
      mu_envelope_set_date (envelope, message_date, msg);
      msg->envelope = envelope;
    }
  *penvelope = msg->envelope;
  return 0;
}

int
mu_message_set_envelope (mu_message_t msg, mu_envelope_t envelope, void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  if (msg->envelope)
    mu_envelope_destroy (&(msg->envelope), msg);
  msg->envelope = envelope;
  msg->flags |= MESSAGE_MODIFIED;
  return 0;
}

int
mu_message_get_attribute (mu_message_t msg, mu_attribute_t *pattribute)
{
  if (msg == NULL)
    return EINVAL;
  if (pattribute == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (msg->attribute == NULL)
    {
      mu_attribute_t attribute;
      int status = mu_attribute_create (&attribute, msg);
      if (status != 0)
	return status;
      msg->attribute = attribute;
    }
  *pattribute = msg->attribute;
  return 0;
}

int
mu_message_set_attribute (mu_message_t msg, mu_attribute_t attribute, void *owner)
{
  if (msg == NULL)
   return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  if (msg->attribute)
    mu_attribute_destroy (&(msg->attribute), owner);
  msg->attribute = attribute;
  msg->flags |= MESSAGE_MODIFIED;
  return 0;
}

int
mu_message_get_uid (mu_message_t msg, size_t *puid)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->_get_uid)
    return msg->_get_uid (msg, puid);
  *puid = 0;
  return 0;
}

int
mu_message_get_uidl (mu_message_t msg, char *buffer, size_t buflen,
		     size_t *pwriten)
{
  mu_header_t header = NULL;
  size_t n = 0;
  int status;

  if (msg == NULL || buffer == NULL || buflen == 0)
    return EINVAL;

  buffer[0] = '\0';
  /* Try the function overload if error fallback.  */
  if (msg->_get_uidl)
    {
      status = msg->_get_uidl (msg, buffer, buflen, pwriten);
      if (status == 0)
	return status;
    }

  /* Be compatible with Qpopper ? qppoper saves the UIDL in "X-UIDL".
     We generate a chksum and save it in the header.  */
  mu_message_get_header (msg, &header);
  status = mu_header_get_value_unfold (header, "X-UIDL", buffer, buflen, &n);
  if (status != 0 || n == 0)
    {
      size_t uid = 0;
      struct mu_md5_ctx md5context;
      mu_stream_t stream = NULL;
      char buf[1024];
      mu_off_t offset = 0;
      unsigned char md5digest[16];
      char *tmp;
      n = 0;
      mu_message_get_uid (msg, &uid);
      mu_message_get_stream (msg, &stream);
      mu_md5_init_ctx (&md5context);
      while (mu_stream_read (stream, buf, sizeof (buf), offset, &n) == 0
	     && n > 0)
	{
	  mu_md5_process_bytes (buf, n, &md5context);
	  offset += n;
	}
      mu_md5_finish_ctx (&md5context, md5digest);
      tmp = buf;
      for (n = 0; n < 16; n++, tmp += 2)
	sprintf (tmp, "%02x", md5digest[n]);
      *tmp = '\0';
      /* POP3 rfc says that an UID should not be longer than 70.  */
      snprintf (buf + 32, 70, ".%lu.%lu", (unsigned long)time (NULL), 
                (unsigned long) uid);

      mu_header_set_value (header, "X-UIDL", buf, 1);
      buflen--; /* leave space for the NULL.  */
      strncpy (buffer, buf, buflen)[buflen] = '\0';
      status = 0;
    }
  return status;
}

int
mu_message_get_qid (mu_message_t msg, mu_message_qid_t *pqid)
{
  if (msg == NULL)
    return EINVAL;
  if (!msg->_get_qid)
    return ENOSYS;
  return msg->_get_qid (msg, pqid);
}
    
int
mu_message_set_qid (mu_message_t msg,
		    int (*_get_qid) (mu_message_t, mu_message_qid_t *),
		    void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_get_qid = _get_qid;
  return 0;
}

int
mu_message_set_uid (mu_message_t msg, int (*_get_uid) (mu_message_t, size_t *),
		    void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_get_uid = _get_uid;
  return 0;
}

int
mu_message_set_uidl (mu_message_t msg,
		  int (* _get_uidl) (mu_message_t, char *, size_t, size_t *),
		  void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_get_uidl = _get_uidl;
  return 0;
}

int
mu_message_set_is_multipart (mu_message_t msg,
			  int (*_is_multipart) (mu_message_t, int *),
			  void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_is_multipart = _is_multipart;
  return 0;
}

int
mu_message_is_multipart (mu_message_t msg, int *pmulti)
{
  if (msg && pmulti)
    {
      if (msg->_is_multipart)
	return msg->_is_multipart (msg, pmulti);
      if (msg->mime == NULL)
	{
	  int status = mu_mime_create (&(msg->mime), msg, 0);
	  if (status != 0)
	    return 0;
	}
      *pmulti = mu_mime_is_multipart(msg->mime);
    }
  return 0;
}

int
mu_message_get_num_parts (mu_message_t msg, size_t *pparts)
{
  if (msg == NULL || pparts == NULL)
    return EINVAL;

  if (msg->_get_num_parts)
    return msg->_get_num_parts (msg, pparts);

  if (msg->mime == NULL)
    {
      int status = mu_mime_create (&(msg->mime), msg, 0);
      if (status != 0)
	return status;
    }
  return mu_mime_get_num_parts (msg->mime, pparts);
}

int
mu_message_set_get_num_parts (mu_message_t msg,
			   int (*_get_num_parts) (mu_message_t, size_t *),
			   void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_get_num_parts = _get_num_parts;
  return 0;
}

int
mu_message_get_part (mu_message_t msg, size_t part, mu_message_t *pmsg)
{
  if (msg == NULL || pmsg == NULL)
    return EINVAL;

  /* Overload.  */
  if (msg->_get_part)
    return msg->_get_part (msg, part, pmsg);

  if (msg->mime == NULL)
    {
      int status = mu_mime_create (&(msg->mime), msg, 0);
      if (status != 0)
	return status;
    }
  return mu_mime_get_part (msg->mime, part, pmsg);
}

int
mu_message_set_get_part (mu_message_t msg, int (*_get_part)
		      (mu_message_t, size_t, mu_message_t *),
		      void *owner)
{
  if (msg == NULL)
    return EINVAL;
  if (msg->owner != owner)
    return EACCES;
  msg->_get_part = _get_part;
  return 0;
}

int
mu_message_get_observable (mu_message_t msg, mu_observable_t *pobservable)
{
  if (msg == NULL || pobservable == NULL)
    return EINVAL;

  if (msg->observable == NULL)
    {
      int status = mu_observable_create (&(msg->observable), msg);
      if (status != 0)
	return status;
    }
  *pobservable = msg->observable;
  return 0;
}

/* Implements the mu_stream_read () on the message stream.  */
static int
message_read (mu_stream_t is, char *buf, size_t buflen,
	      mu_off_t off, size_t *pnread )
{
  mu_message_t msg =  mu_stream_get_owner (is);
  mu_stream_t his, bis;
  size_t hread, hsize, bread, bsize;

  if (msg == NULL)
    return EINVAL;

  bsize = hsize = bread = hread = 0;
  his = bis = NULL;

  mu_header_size (msg->header, &hsize);
  mu_body_size (msg->body, &bsize);

  /* On some remote sever (POP) the size of the header and body is not known
     until you start reading them.  So by checking hsize == bsize == 0,
     this kludge is a way of detecting the anomalie and start by the
     header.  */
  if ((size_t)off < hsize || (hsize == 0 && bsize == 0))
    {
      mu_header_get_stream (msg->header, &his);
      mu_stream_read (his, buf, buflen, off, &hread);
    }
  else
    {
      mu_body_get_stream (msg->body, &bis);
      mu_stream_read (bis, buf, buflen, off - hsize, &bread);
    }

  if (pnread)
    *pnread = hread + bread;
  return 0;
}

/* Implements the mu_stream_write () on the message stream.  */
static int
message_write (mu_stream_t os, const char *buf, size_t buflen,
	       mu_off_t off, size_t *pnwrite)
{
  mu_message_t msg = mu_stream_get_owner (os);
  int status = 0;
  size_t bufsize = buflen;

  if (msg == NULL)
    return EINVAL;

  /* Skip the obvious.  */
  if (buf == NULL || buflen == 0)
    {
      if (pnwrite)
	*pnwrite = 0;
      return 0;
    }

  if (!msg->hdr_done)
    {
      size_t len;
      char *nl;
      mu_header_t header = NULL;
      mu_stream_t hstream = NULL;
      mu_message_get_header (msg, &header);
      mu_header_get_stream (header, &hstream);
      while (!msg->hdr_done && (nl = memchr (buf, '\n', buflen)) != NULL)
	{
	  len = nl - buf + 1;
	  status = mu_stream_write (hstream, buf, len, msg->hdr_buflen, NULL);
	  if (status != 0)
	    return status;
	  msg->hdr_buflen += len;
	  /* We detect an empty line .i.e "^\n$" this signal the end of the
	     header.  */
	  if (buf == nl)
	    msg->hdr_done = 1;
	  buf = nl + 1;
	  buflen -= len;
	}
    }

  /* Message header is not complete but was not a full line.  */
  if (!msg->hdr_done && buflen > 0)
    {
      mu_header_t header = NULL;
      mu_stream_t hstream = NULL;
      mu_message_get_header (msg, &header);
      mu_header_get_stream (header, &hstream);
      status = mu_stream_write (hstream, buf, buflen, msg->hdr_buflen, NULL);
      if (status != 0)
	return status;
      msg->hdr_buflen += buflen;
      buflen = 0;
    }
  else if (buflen > 0) /* In the body.  */
    {
      mu_stream_t bs;
      mu_body_t body;
      size_t written = 0;
      if ((status = mu_message_get_body (msg, &body)) != 0 ||
	  (status = mu_body_get_stream (msg->body, &bs)) != 0)
	{
	  msg->hdr_buflen = msg->hdr_done = 0;
	  return status;
	}
      if (off < (mu_off_t)msg->hdr_buflen)
	off = 0;
      else
	off -= msg->hdr_buflen;
      status = mu_stream_write (bs, buf, buflen, off, &written);
      buflen -= written;
    }
  if (pnwrite)
    *pnwrite = bufsize - buflen;
  return status;
}

static int
message_get_transport2 (mu_stream_t stream, mu_transport_t *pin,
			mu_transport_t *pout)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  mu_body_t body;
  mu_stream_t is;

  if (msg == NULL)
    return EINVAL;
  if (pout)
    *pout = NULL;

  /* Probably being lazy, then create a body for the stream.  */
  if (msg->body == NULL)
    {
      int status = mu_body_create (&body, msg);
      if (status != 0 )
	return status;
      msg->body = body;
    }
  else
      body = msg->body;

  mu_body_get_stream (body, &is);
  return mu_stream_get_transport2 (is, pin, pout);
}

/* Implements the stream_stream_size () on the message stream.  */
static int
message_stream_size (mu_stream_t stream, mu_off_t *psize)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  size_t size;
  int rc = mu_message_size (msg, &size); /* FIXME: should it get mu_off_t as
                                            its 2nd argument */
  if (rc == 0)
    *psize = size;
  return rc;
}

static int
message_date (mu_envelope_t envelope, char *buf, size_t len, size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  time_t t;
  size_t n;

  if (msg == NULL)
    return EINVAL;

  /* FIXME: extract the time from "Date:".  */

  if (buf == NULL || len == 0)
    {
      n = MU_ENVELOPE_DATE_LENGTH;
    }
  else
    {
      char tmpbuf[MU_ENVELOPE_DATE_LENGTH+1];
      t = time (NULL);
      n = mu_strftime (tmpbuf, sizeof tmpbuf, 
                       MU_ENVELOPE_DATE_FORMAT, localtime (&t));
      n = mu_cpystr (buf, tmpbuf, len);
    }
  if (pnwrite)
    *pnwrite = n;
  return 0;
}

static int
message_sender (mu_envelope_t envelope, char *buf, size_t len, size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  mu_header_t header = NULL;
  size_t n = 0;
  int status;

  if (msg == NULL)
    return EINVAL;

  /* Can it be extracted from the From:  */
  mu_message_get_header (msg, &header);
  status = mu_header_get_value (header, MU_HEADER_FROM, NULL, 0, &n);
  if (status == 0 && n != 0)
    {
      char *sender;
      mu_address_t address = NULL;
      sender = calloc (1, n + 1);
      if (sender == NULL)
	return ENOMEM;
      mu_header_get_value (header, MU_HEADER_FROM, sender, n + 1, NULL);
      if (mu_address_create (&address, sender) == 0)
	mu_address_get_email (address, 1, buf, n + 1, pnwrite);
      free (sender);
      mu_address_destroy (&address);
      return 0;
    }
  else if (status == EAGAIN)
    return status;

  /* oops! We are still here */
  {
    struct mu_auth_data *auth = mu_get_auth_by_uid (getuid ());
    const char *sender = auth ? auth->name : "unknown";
    n = strlen (sender);
    if (buf && len > 0)
      {
	len--; /* One for the null.  */
	n = (n < len) ? n : len;
	memcpy (buf, auth->name, n);
	buf[n] = '\0';
      }
    if (auth)
      mu_auth_data_free (auth);
  }

  if (pnwrite)
    *pnwrite = n;
  return 0;
}

static int
message_header_fill (mu_header_t header, char *buffer, size_t buflen,
		     mu_off_t off, size_t * pnread)
{
  int status = 0;
  mu_message_t msg = mu_header_get_owner (header);
  mu_stream_t stream = NULL;
  size_t nread = 0;

  /* Noop.  */
  if (buffer == NULL || buflen == 0)
    {
      if (pnread)
        *pnread = nread;
      return 0;
    }

  if (!msg->hdr_done)
    {
      status = mu_message_get_stream (msg, &stream);
      if (status == 0)
	{
	  /* Position the file pointer and the buffer.  */
	  status = mu_stream_readline (stream, buffer, buflen, off, &nread);
	  /* Detect the end of the headers. */
	  if (nread  && buffer[0] == '\n' && buffer[1] == '\0')
	    {
	      msg->hdr_done = 1;
	    }
	  msg->hdr_buflen += nread;
	}
    }

  if (pnread)
    *pnread = nread;

  return status;
}

static int
message_body_read (mu_stream_t stream,  char *buffer, size_t n, mu_off_t off,
		   size_t *pn)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  size_t nread = 0;
  mu_header_t header = NULL;
  mu_stream_t bstream = NULL;
  size_t size = 0;
  int status;

  mu_message_get_header (msg, &header);
  status = mu_header_size (msg->header, &size);
  if (status == 0)
    {
      mu_message_get_stream (msg, &bstream);
      status = mu_stream_read (bstream, buffer, n, size + off, &nread);
    }
  if (pn)
    *pn = nread;
  return status;
}

int
mu_message_save_to_mailbox (mu_message_t msg, 
                            mu_debug_t debug,
			    const char *toname, int perms)
{
  int rc = 0;
  mu_mailbox_t to = 0;

  if ((rc = mu_mailbox_create_default (&to, toname)))
    {
      MU_DEBUG2 (debug, MU_DEBUG_ERROR,
		 "mu_mailbox_create_default (%s) failed: %s\n", toname,
		 mu_strerror (rc));
      goto end;
    }

  if (debug && (rc = mu_mailbox_set_debug (to, debug)))
	goto end;

  if ((rc = mu_mailbox_open (to,
			     MU_STREAM_WRITE | MU_STREAM_CREAT
			     | (perms & MU_STREAM_IMASK))))
    {
      MU_DEBUG2 (debug, MU_DEBUG_ERROR,
		 "mu_mailbox_open (%s) failed: %s\n", toname,
		 mu_strerror (rc));
      goto end;
    }

  if ((rc = mu_mailbox_append_message (to, msg)))
    {
      MU_DEBUG2 (debug, MU_DEBUG_ERROR,
		 "mu_mailbox_append_message (%s) failed: %s\n", toname,
		 mu_strerror (rc));
      goto end;
    }

end:

  if (!rc)
    {
      if ((rc = mu_mailbox_close (to)))
	MU_DEBUG2 (debug, MU_DEBUG_ERROR,
		   "mu_mailbox_close (%s) failed: %s\n", toname,
		   mu_strerror (rc));
    }
  else
    mu_mailbox_close (to);

  mu_mailbox_destroy (&to);

  return rc;
}

