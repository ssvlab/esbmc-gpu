/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#ifdef ENABLE_NNTP

#include <termios.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdarg.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/md5.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/message.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/iterator.h>
#include <mailutils/url.h>
#include <mailutils/nntp.h>

#include <folder0.h>
#include <mailbox0.h>
#include "nntp0.h"


/*  Functions/Methods that implements the mu_mailbox_t API.  */
static void nntp_mailbox_destroy         (mu_mailbox_t);
static int  nntp_mailbox_open            (mu_mailbox_t, int);
static int  nntp_mailbox_close           (mu_mailbox_t);
static int  nntp_mailbox_get_message     (mu_mailbox_t, size_t, mu_message_t *);
static int  nntp_mailbox_messages_count  (mu_mailbox_t, size_t *);
static int  nntp_mailbox_scan            (mu_mailbox_t, size_t, size_t *);
/* FIXME
   static int  nntp_mailbox_get_size        (mu_mailbox_t, mu_off_t *); */

static int  nntp_message_get_transport2  (mu_stream_t, mu_transport_t *, mu_transport_t *);
static int  nntp_message_read            (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int  nntp_message_size            (mu_message_t, size_t *);
/* FIXME
   static int  nntp_message_line            (mu_message_t, size_t *); */
static int  nntp_message_uidl            (mu_message_t, char *, size_t, size_t *);
static int  nntp_message_uid             (mu_message_t, size_t *);

/* FIXME
   static int  nntp_header_get_transport2   (mu_header_t, char *,
                                             size_t, mu_off_t, size_t *); */
static int  nntp_header_fill             (mu_header_t, char *, size_t, mu_off_t, size_t *);

static int  nntp_body_get_transport2     (mu_stream_t, mu_transport_t *, mu_transport_t *);
static int  nntp_body_read               (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int  nntp_body_size               (mu_body_t, size_t *);
static int  nntp_body_lines              (mu_body_t, size_t *);

static int  nntp_get_transport2          (msg_nntp_t, mu_transport_t *, mu_transport_t *);

int
_nntp_mailbox_init (mu_mailbox_t mbox)
{
  m_nntp_t m_nntp;
  int status = 0;

  /* Allocate specifics for nntp data.  */
  m_nntp = mbox->data = calloc (1, sizeof (*m_nntp));
  if (mbox->data == NULL)
    return ENOMEM;

  /* Get the back pointer of the concrete folder. */
  if (mbox->folder)
    m_nntp->f_nntp = mbox->folder->data;

  m_nntp->mailbox = mbox;		/* Back pointer.  */

  /* Retrieve the name of the newsgroup from the URL.  */
  status = mu_url_aget_path (mbox->url, &m_nntp->name);
  if (status == MU_ERR_NOENT)
    {
      m_nntp->name = strdup ("INBOX");
      if (!m_nntp->name)
        return ENOMEM;
    }
  else if (status)
    return status;
  else  
    {
      char *p;
      p = strchr (m_nntp->name, '/');
      if (p)
	*p = '\0';
    }

  /* Initialize the structure.  */
  mbox->_destroy = nntp_mailbox_destroy;

  mbox->_open = nntp_mailbox_open;
  mbox->_close = nntp_mailbox_close;

  /* Messages.  */
  mbox->_get_message = nntp_mailbox_get_message;
  mbox->_messages_count = nntp_mailbox_messages_count;
  mbox->_messages_recent = nntp_mailbox_messages_count;
  mbox->_message_unseen = nntp_mailbox_messages_count;
  /*mbox->_expunge = nntp_mailbox_expunge;*/

  mbox->_scan = nntp_mailbox_scan;
  /*mbox->_is_updated = nntp_mailbox_is_updated; */

  /*mbox->_get_size = nntp_mailbox_get_size; */

  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailbox_get_property (mbox, &property);
    mu_property_set_value (property, "TYPE", "NNTP", 1);
  }

  return status;
}

/*  Cleaning up all the ressources associate with a newsgroup/mailbox.  */
static void
nntp_mailbox_destroy (mu_mailbox_t mbox)
{
  if (mbox->data)
    {
      m_nntp_t m_nntp = mbox->data;
      f_nntp_t f_nntp = m_nntp->f_nntp;
      size_t i;

      /* Deselect.  */
      if (m_nntp == f_nntp->selected)
	f_nntp->selected = NULL;

      mu_monitor_wrlock (mbox->monitor);

      if (m_nntp->name)
	free (m_nntp->name);

      /* Destroy the nntp messages and ressources associated to them.  */
      for (i = 0; i < m_nntp->messages_count; i++)
	{
	  if (m_nntp->messages[i])
	    {
	      mu_message_destroy (&(m_nntp->messages[i]->message), m_nntp->messages[i]);
	      if (m_nntp->messages[i]->mid)
		free (m_nntp->messages[i]->mid);
	      free (m_nntp->messages[i]);
	      m_nntp->messages[i] = NULL;
	    }
	}
      if (m_nntp->messages)
	free (m_nntp->messages);
      free (m_nntp);
      mbox->data = NULL;
      mu_monitor_unlock (mbox->monitor);
    }
}

/* If the connection was not up it is open by the folder since the stream
   socket is actually created by the folder.  It is not necessary
   to set select the mailbox/newsgoup right away, there are maybe on going operations.
   But on any operation by a particular mailbox, it will be selected first.  */
static int
nntp_mailbox_open (mu_mailbox_t mbox, int flags)
{
  int status = 0;
  m_nntp_t m_nntp = mbox->data;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  mu_iterator_t iterator;

  /* m_nntp must have been created during mailbox initialization. */
  /* assert (mbox->data);
     assert (m_nntp->name); */

  mbox->flags = flags;

  /* make sure the connection is up.  */
  if ((status = mu_folder_open (f_nntp->folder, flags)))
    return status;

  mu_nntp_set_debug (f_nntp->nntp, mbox->debug);

  /* We might not have to SELECT the newsgroup, but we need to know it
     exists.  */
  status = mu_nntp_list_active (f_nntp->nntp, m_nntp->name, &iterator);
  if (status == 0)
    {
      for (mu_iterator_first (iterator);
           !mu_iterator_is_done (iterator); mu_iterator_next (iterator))
        {
          char *buffer = NULL;
          mu_iterator_current (iterator, (void **) &buffer);
          mu_nntp_parse_list_active (buffer, NULL, &m_nntp->high, &m_nntp->low, &m_nntp->status);
        }
      mu_iterator_destroy (&iterator);
    }
  return status;
}

/* We can not close the folder in term of shuting down the connection but if
   we were the selected mailbox/newsgroup we deselect ourself.  */
static int
nntp_mailbox_close (mu_mailbox_t mailbox)
{
  m_nntp_t m_nntp = mailbox->data;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  int i;

  mu_monitor_wrlock (mailbox->monitor);

  /* Destroy the nntp posts and ressources associated to them.  */
  for (i = 0; i < m_nntp->messages_count; i++)
    {
      if (m_nntp->messages[i])
	{
	  msg_nntp_t msg_nntp = m_nntp->messages[i];
	  if (msg_nntp->message)
	    mu_message_destroy (&(msg_nntp->message), msg_nntp);
	}
      free (m_nntp->messages[i]);
    }
  if (m_nntp->messages)
    free (m_nntp->messages);
  m_nntp->messages = NULL;
  m_nntp->messages_count = 0;
  m_nntp->number = 0;
  m_nntp->low = 0;
  m_nntp->high = 0;
  mu_monitor_unlock (mailbox->monitor);

  /* Deselect.  */
  if (m_nntp != f_nntp->selected)
    f_nntp->selected = NULL;

  /* Decrement the ref count. */
  return mu_folder_close (mailbox->folder);
}

static int
nntp_mailbox_get_message (mu_mailbox_t mbox, size_t msgno, mu_message_t *pmsg)
{
  m_nntp_t m_nntp = mbox->data;
  msg_nntp_t msg_nntp;
  mu_message_t msg = NULL;
  int status;
  size_t i;

  /* Sanity.  */
 if (pmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;

 msgno--;
  mu_monitor_rdlock (mbox->monitor);
  /* See if we have already this message.  */
  for (i = 0; i < m_nntp->messages_count; i++)
    {
      if (m_nntp->messages[i])
	{
	  if (m_nntp->messages[i]->msgno == msgno + m_nntp->low)
	    {
	      *pmsg = m_nntp->messages[i]->message;
	      mu_monitor_unlock (mbox->monitor);
	      return 0;
	    }
	}
    }
  mu_monitor_unlock (mbox->monitor);

  msg_nntp = calloc (1, sizeof (*msg_nntp));
  if (msg_nntp == NULL)
    return ENOMEM;

  /* Back pointer.  */
  msg_nntp->m_nntp = m_nntp;
  msg_nntp->msgno = msgno + m_nntp->low;

  /* Create the message.  */
  {
    mu_stream_t stream = NULL;
    if ((status = mu_message_create (&msg, msg_nntp)) != 0
	|| (status = mu_stream_create (&stream, mbox->flags, msg)) != 0)
      {
	mu_stream_destroy (&stream, msg);
	mu_message_destroy (&msg, msg_nntp);
	free (msg_nntp);
	return status;
      }
    /* Help for the readline()s  */
    mu_stream_set_read (stream, nntp_message_read, msg);
    mu_stream_set_get_transport2 (stream, nntp_message_get_transport2, msg);
    mu_message_set_stream (msg, stream, msg_nntp);
    mu_message_set_size (msg, nntp_message_size, msg_nntp);
  }

  /* Create the header.  */
  {
    mu_header_t header = NULL;
    if ((status = mu_header_create (&header, NULL, 0,  msg)) != 0)
      {
	mu_message_destroy (&msg, msg_nntp);
	free (msg_nntp);
	return status;
      }
    mu_header_set_fill (header, nntp_header_fill, msg);
    mu_message_set_header (msg, header, msg_nntp);
  }

  /* Create the body and its stream.  */
  {
    mu_body_t body = NULL;
    mu_stream_t stream = NULL;
    if ((status = mu_body_create (&body, msg)) != 0
	|| (status = mu_stream_create (&stream, mbox->flags, body)) != 0)
      {
	mu_body_destroy (&body, msg);
	mu_stream_destroy (&stream, body);
	mu_message_destroy (&msg, msg_nntp);
	free (msg_nntp);
	return status;
      }
    /* Helps for the readline()s  */
    mu_stream_set_read (stream, nntp_body_read, body);
    mu_stream_set_get_transport2 (stream, nntp_body_get_transport2, body);
    mu_body_set_size (body, nntp_body_size, msg);
    mu_body_set_lines (body, nntp_body_lines, msg);
    mu_body_set_stream (body, stream, msg);
    mu_message_set_body (msg, body, msg_nntp);
  }

  /* Set the UID on the message. */
  mu_message_set_uid (msg, nntp_message_uid, msg_nntp);

  /* Add it to the list.  */
  mu_monitor_wrlock (mbox->monitor);
  {
    msg_nntp_t *m ;
    m = realloc (m_nntp->messages, (m_nntp->messages_count + 1)*sizeof (*m));
    if (m == NULL)
      {
	mu_message_destroy (&msg, msg_nntp);
	free (msg_nntp);
	mu_monitor_unlock (mbox->monitor);
	return ENOMEM;
      }
    m_nntp->messages = m;
    m_nntp->messages[m_nntp->messages_count] = msg_nntp;
    m_nntp->messages_count++;
  }
  mu_monitor_unlock (mbox->monitor);

  /* Save The message pointer.  */
  mu_message_set_mailbox (msg, mbox, msg_nntp);
  *pmsg = msg_nntp->message = msg;

  return 0;
}

/* There is no explicit call to get the message count.  The count is send on
   a "GROUP" command.  The function is also use as a way to select newsgoupr by other functions.  */
static int
nntp_mailbox_messages_count (mu_mailbox_t mbox, size_t *pcount)
{
  m_nntp_t m_nntp = mbox->data;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  int status = 0;

  status = mu_folder_open (mbox->folder, mbox->flags);
  if (status != 0)
    return status;

  /* Are we already selected ? */
  if (m_nntp == (f_nntp->selected))
    {
      if (pcount)
        *pcount = m_nntp->number;
      return 0;
    }

  /*  Put the mailbox as selected.  */
  f_nntp->selected = m_nntp;

  status = mu_nntp_group (f_nntp->nntp, m_nntp->name, &m_nntp->number, &m_nntp->low, &m_nntp->high, NULL);

  if (pcount)
    *pcount = m_nntp->number;

  return status;
}

/* Update and scanning. FIXME: Is not used */
static int
nntp_is_updated (mu_mailbox_t mbox)
{
  return 1;
}

/* We just simulate by sending a notification for the total msgno.  */
/* FIXME is message is set deleted should we sent a notif ?  */
static int
nntp_mailbox_scan (mu_mailbox_t mbox, size_t msgno, size_t *pcount)
{
  int status;
  size_t i;
  size_t count = 0;

  /* Select first.  */
  status = nntp_mailbox_messages_count (mbox, &count);
  if (pcount)
    *pcount = count;
  if (status != 0)
    return status;
  if (mbox->observable == NULL)
    return 0;
  for (i = msgno; i <= count; i++)
    {
      size_t tmp = i;
      if (mu_observable_notify (mbox->observable, MU_EVT_MESSAGE_ADD,
				&tmp) != 0)
	break;
      if ((i +1) % 10 == 0)
	mu_observable_notify (mbox->observable, MU_EVT_MAILBOX_PROGRESS, NULL);
    }
  return 0;
}

static int
nntp_message_size (mu_message_t msg, size_t *psize)
{
  if (psize)
    *psize = 0;
  return 0;
}

static int
nntp_body_size (mu_body_t body, size_t *psize)
{
  if (psize)
    *psize = 0;

  return 0;
}

/* Not know until the whole message get downloaded.  */
static int
nntp_body_lines (mu_body_t body, size_t *plines)
{
  if (plines)
    *plines = 0;
  return 0;
}

/* Stub to call the fd from body object.  */
static int
nntp_body_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  return nntp_get_transport2 (msg_nntp, pin, pout);
}

/* Stub to call the fd from message object.  */
static int
nntp_message_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  return nntp_get_transport2 (msg_nntp, pin, pout);
}

static int
nntp_get_transport2 (msg_nntp_t msg_nntp, mu_transport_t *pin, mu_transport_t *pout)
{
  int status = EINVAL;
  if (msg_nntp && msg_nntp->m_nntp
      && msg_nntp->m_nntp->f_nntp && msg_nntp->m_nntp->f_nntp->folder)
    {
      mu_stream_t carrier;
      status = mu_nntp_get_carrier (msg_nntp->m_nntp->f_nntp->nntp, &carrier);
      if (status == 0)
	return mu_stream_get_transport2 (carrier, pin, pout);
    }
  return status;
}

static int
nntp_message_uid (mu_message_t msg,  size_t *puid)
{
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  m_nntp_t m_nntp = msg_nntp->m_nntp;
  int status;

  if (puid)
    return 0;

  /* Select first.  */
  status = nntp_mailbox_messages_count (m_nntp->mailbox, NULL);
  if (status != 0)
    return status;

  if (puid)
    *puid = msg_nntp->msgno;
  return 0;
}

static int
nntp_message_uidl (mu_message_t msg, char *buffer, size_t buflen,
		   size_t *pnwriten)
{
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  m_nntp_t m_nntp = msg_nntp->m_nntp;
  int status = 0;

  /* Select first.  */
  status = nntp_mailbox_messages_count (m_nntp->mailbox, NULL);
  if (status != 0)
    return status;

  if (msg_nntp->mid)
    {
      size_t len = strlen (msg_nntp->mid);
      if (buffer)
	{
	  buflen--; /* Leave space for the null.  */
	  buflen = (len > buflen) ? buflen : len;
	  memcpy (buffer, msg_nntp->mid, buflen);
	  buffer[buflen] = '\0';
	}
      else
	buflen = len;
    }
  else
    buflen = 0;

  if (pnwriten)
    *pnwriten = buflen;
  return status;
}

/* Message read overload  */
static int
nntp_message_read (mu_stream_t stream, char *buffer, size_t buflen, mu_off_t offset, size_t *plen)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  m_nntp_t m_nntp = msg_nntp->m_nntp;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  int status;
  size_t len = 0;

  /* Start over.  */
  if (plen == NULL)
    plen = &len;

  /* Select first.  */
  status = nntp_mailbox_messages_count (m_nntp->mailbox, NULL);
  if (status != 0)
    return status;

  if (msg_nntp->mstream == NULL)
    {
      status = mu_nntp_article (f_nntp->nntp, msg_nntp->msgno, NULL, &msg_nntp->mid,  &msg_nntp->mstream);
      if (status != 0)
	return status;
    }
  status = mu_stream_read (msg_nntp->mstream, buffer, buflen, offset, plen);
  if (status == 0)
    {
      /* Destroy the stream.  */
      if (*plen == 0)
	{
	  mu_stream_destroy (&msg_nntp->mstream, NULL);
	}
    }
  return status;
}

/* Message read overload  */
static int
nntp_body_read (mu_stream_t stream, char *buffer, size_t buflen, mu_off_t offset, size_t *plen)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  m_nntp_t m_nntp = msg_nntp->m_nntp;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  int status;
  size_t len = 0;

  /* Start over.  */
  if (plen == NULL)
    plen = &len;

  /* Select first.  */
  status = nntp_mailbox_messages_count (m_nntp->mailbox, NULL);
  if (status != 0)
    return status;

  if (msg_nntp->bstream == NULL)
    {
      status = mu_nntp_body (f_nntp->nntp, msg_nntp->msgno, NULL, &msg_nntp->mid,  &msg_nntp->bstream);
      if (status != 0)
	return status;
    }
  status = mu_stream_read (msg_nntp->bstream, buffer, buflen, offset, plen);
  if (status == 0)
    {
      /* Destroy the stream.  */
      if (*plen == 0)
	{
	  mu_stream_destroy (&msg_nntp->bstream, NULL);
	}
    }
  return status;
}

/* Header read overload  */
static int
nntp_header_fill (mu_header_t header, char *buffer, size_t buflen, mu_off_t offset, size_t *plen)
{
  mu_message_t msg = mu_header_get_owner (header);
  msg_nntp_t msg_nntp = mu_message_get_owner (msg);
  m_nntp_t m_nntp = msg_nntp->m_nntp;
  f_nntp_t f_nntp = m_nntp->f_nntp;
  int status;
  size_t len = 0;

  /* Start over.  */
  if (plen == NULL)
    plen = &len;

  /* Select first.  */
  status = nntp_mailbox_messages_count (m_nntp->mailbox, NULL);
  if (status != 0)
    return status;

  if (msg_nntp->hstream == NULL)
    {
      status = mu_nntp_head (f_nntp->nntp, msg_nntp->msgno, NULL, &msg_nntp->mid,  &msg_nntp->hstream);
      if (status != 0)
	return status;
    }
  status = mu_stream_read (msg_nntp->hstream, buffer, buflen, offset, plen);
  if (status == 0)
    {
      /* Destroy the stream.  */
      if (*plen == 0)
	{
	  mu_stream_destroy (&msg_nntp->hstream, NULL);
	}
    }
  return status;
}

#endif
