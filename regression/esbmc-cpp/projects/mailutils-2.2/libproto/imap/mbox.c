/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2006, 2007, 2009,
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

#ifdef ENABLE_IMAP

#include <errno.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <mailutils/address.h>
#include <mailutils/attribute.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/envelope.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/header.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/io.h>

#include <imap0.h>
#include <mailbox0.h>
#include <registrar0.h>
#include <url0.h>

#undef min
#define min(a,b) ((a) < (b) ? (a) : (b))

#define MU_IMAP_CACHE_HEADERS "Bcc Cc Content-Language Content-Transfer-Encoding Content-Type Date From In-Reply-To Message-ID Reference Reply-To Sender Subject To X-UIDL"

/* mu_mailbox_t API.  */
static void mailbox_imap_destroy  (mu_mailbox_t);
static int  mailbox_imap_open     (mu_mailbox_t, int);
static int  mailbox_imap_close    (mu_mailbox_t);
static int  imap_uidvalidity      (mu_mailbox_t, unsigned long *);
static int  imap_uidnext          (mu_mailbox_t, size_t *);
static int  imap_expunge          (mu_mailbox_t);
static int  imap_get_message      (mu_mailbox_t, size_t, mu_message_t *);
static int  imap_messages_count   (mu_mailbox_t, size_t *);
static int  imap_messages_recent  (mu_mailbox_t, size_t *);
static int  imap_message_unseen   (mu_mailbox_t, size_t *);
static int  imap_scan             (mu_mailbox_t, size_t, size_t *);
static int  imap_scan0            (mu_mailbox_t, size_t, size_t *, int);
static int  imap_is_updated       (mu_mailbox_t);
static int  imap_append_message   (mu_mailbox_t, mu_message_t);
static int  imap_append_message0  (mu_mailbox_t, mu_message_t);
static int  imap_copy_message     (mu_mailbox_t, mu_message_t);

/* mu_message_t API.  */
static int  imap_submessage_size  (msg_imap_t, size_t *);
static int  imap_message_size     (mu_message_t, size_t *);
static int  imap_message_lines    (mu_message_t, size_t *);
static int  imap_message_get_transport2  (mu_stream_t, mu_transport_t *pin, 
                                          mu_transport_t *pout);
static int  imap_message_read     (mu_stream_t , char *, size_t, mu_off_t, size_t *);
static int  imap_message_uid      (mu_message_t, size_t *);

/* mu_mime_t API.  */
static int  imap_is_multipart     (mu_message_t, int *);
static int  imap_get_num_parts    (mu_message_t, size_t *);
static int  imap_get_part         (mu_message_t, size_t, mu_message_t *);

/* mu_envelope_t API  */
static int  imap_envelope_sender  (mu_envelope_t, char *, size_t, size_t *);
static int  imap_envelope_date    (mu_envelope_t, char *, size_t, size_t *);

/* mu_attribute_t API  */
static int  imap_attr_get_flags   (mu_attribute_t, int *);
static int  imap_attr_set_flags   (mu_attribute_t, int);
static int  imap_attr_unset_flags (mu_attribute_t, int);

/* mu_header_t API.  */
static int  imap_header_read      (mu_header_t, char*, size_t, mu_off_t, size_t *);

/* mu_body_t API.  */
static int  imap_body_read        (mu_stream_t, char *, size_t, mu_off_t, size_t *);
static int  imap_body_size        (mu_body_t, size_t *);
static int  imap_body_lines       (mu_body_t, size_t *);
static int  imap_body_get_transport2 (mu_stream_t, mu_transport_t *pin, mu_transport_t *pout);

/* Helpers.  */
static int  imap_get_transport2    (msg_imap_t msg_imap, 
                                    mu_transport_t *pin,
                                    mu_transport_t *pout);
static int  imap_get_message0     (msg_imap_t, mu_message_t *);
static int  fetch_operation       (f_imap_t, msg_imap_t, char *, size_t, size_t *);
static void free_subparts         (msg_imap_t);
static int  flags_to_string       (char **, int);
static int  delete_to_string      (m_imap_t, char **);
static int  is_same_folder        (mu_mailbox_t, mu_message_t);

#define MBX_WRITABLE(mbx) ((mbx)->flags & (MU_STREAM_WRITE|MU_STREAM_RDWR|MU_STREAM_CREAT))

/* Initialize the concrete object mu_mailbox_t by overloading the function of the
   structure.  */
int
_mailbox_imap_and_imaps_init (mu_mailbox_t mailbox, int imaps)
{
  int status;
  m_imap_t m_imap;

  if (!mailbox)
    return MU_ERR_MBX_NULL;
  if (mailbox->folder == NULL)
    return EINVAL;
  
  m_imap = mailbox->data = calloc (1, sizeof (*m_imap));
  if (m_imap == NULL)
    return ENOMEM;

  /* Retrieve the name of the mailbox from the URL.  */
  status = mu_url_aget_path (mailbox->url, &m_imap->name);
  if (status == MU_ERR_NOENT)
    {
      m_imap->name = strdup ("INBOX");
      if (!m_imap->name)
        return ENOMEM;
    }
  else if (status)
    return status;

  /* Overload the functions.  */
  mailbox->_destroy = mailbox_imap_destroy;

  mailbox->_open = mailbox_imap_open;
  mailbox->_close = mailbox_imap_close;

  /* Messages.  */
  mailbox->_get_message = imap_get_message;
  mailbox->_append_message = imap_append_message;
  mailbox->_messages_count = imap_messages_count;
  mailbox->_messages_recent = imap_messages_recent;
  mailbox->_message_unseen = imap_message_unseen;
  mailbox->_expunge = imap_expunge;
  mailbox->_uidvalidity = imap_uidvalidity;
  mailbox->_uidnext = imap_uidnext;

  mailbox->_scan = imap_scan;
  mailbox->_is_updated = imap_is_updated;

  /* Get the back pointer of the concrete folder. */
  m_imap->f_imap = mailbox->folder->data;
  m_imap->f_imap->imaps = imaps;
  
  /* maibox back pointer.  */
  m_imap->mailbox = mailbox;

  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailbox_get_property (mailbox, &property);
    mu_property_set_value (property, "TYPE", "IMAP4", 1);
  }

  return 0;
}

int
_mailbox_imap_init (mu_mailbox_t mailbox)
{
  return _mailbox_imap_and_imaps_init (mailbox, 0);
}

int
_mailbox_imaps_init (mu_mailbox_t mailbox)
{
  return _mailbox_imap_and_imaps_init (mailbox, 1);
}


/* Recursive call to free all the subparts of a message.  */
static void
free_subparts (msg_imap_t msg_imap)
{
  size_t i;
  for (i = 0; i < msg_imap->num_parts; i++)
    {
      if (msg_imap->parts[i])
	free_subparts (msg_imap->parts[i]);
    }

  if (msg_imap->message)
    mu_message_destroy (&(msg_imap->message), msg_imap);
  if (msg_imap->parts)
    free (msg_imap->parts);
  if (msg_imap->fheader)
    mu_header_destroy (&msg_imap->fheader, NULL);
  if (msg_imap->internal_date)
    free (msg_imap->internal_date);
  free(msg_imap);
}

/* Give back all the resources. But it does not mean to shutdown the channel
   this is done on the folder.  */
static void
mailbox_imap_destroy (mu_mailbox_t mailbox)
{
  if (mailbox->data)
    {
      m_imap_t m_imap = mailbox->data;
      f_imap_t f_imap = m_imap->f_imap;
      size_t i;

      /* Deselect.  */
      if (m_imap != f_imap->selected)
	f_imap->selected = NULL;

      mu_monitor_wrlock (mailbox->monitor);
      /* Destroy the imap messages and ressources associated to them.  */
      for (i = 0; i < m_imap->imessages_count; i++)
	{
	  if (m_imap->imessages[i])
	    free_subparts (m_imap->imessages[i]);
	}
      if (m_imap->imessages)
	free (m_imap->imessages);
      if (m_imap->name)
	free (m_imap->name);
      free (m_imap);
      mailbox->data = NULL;
      mu_monitor_unlock (mailbox->monitor);
    }
}

/* If the connection was not up it is open by the folder since the stream
   socket is actually created by the folder.  It is not necessary
   to set select the mailbox right away, there are maybe on going operations.
   But on any operation by a particular mailbox, it will be selected first.  */
static int
mailbox_imap_open (mu_mailbox_t mailbox, int flags)
{
  int status = 0;
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;
  mu_folder_t folder = f_imap->folder;
  mu_list_t folders = NULL;
  size_t count;
  
  /* m_imap must have been created during mailbox initialization. */
  assert (mailbox->data);
  assert (m_imap->name);

  mailbox->flags = flags;

  if ((status = mu_folder_open (mailbox->folder, flags)))
    return status;

  /* We might not have to SELECT the mailbox, but we need to know it
     exists, and CREATE it if it doesn't, and CREATE is specified in
     the flags.
   */

  switch (m_imap->state)
    {
    case IMAP_NO_STATE:
      m_imap->state = IMAP_LIST;

    case IMAP_LIST:
      status = mu_folder_list (folder, NULL, m_imap->name, 0, &folders);
      if (status != 0)
	{
	  if (status != EAGAIN && status != EINPROGRESS && status != EINTR)
	    m_imap->state = IMAP_NO_STATE;

	  return status;
	}
      m_imap->state = IMAP_NO_STATE;
      status = mu_list_count (folders, &count);
      mu_list_destroy (&folders);
      if (status || count)
	return 0;

      if ((flags & MU_STREAM_CREAT) == 0)
	return ENOENT;

      m_imap->state = IMAP_CREATE;

    case IMAP_CREATE:
      switch (f_imap->state)
	{
	case IMAP_NO_STATE:
	  {
	    const char *path;
	    status = mu_url_sget_path (folder->url, &path);
	    if (status == MU_ERR_NOENT)
	      return 0;
	    else if (status)
	      return status;
	    status = imap_writeline (f_imap, "g%lu CREATE %s\r\n",
				     (unsigned long) f_imap->seq, path);
	    MU_DEBUG2 (folder->debug, MU_DEBUG_PROT, "g%lu CREATE %s\n",
		       (unsigned long) f_imap->seq, path);
	    f_imap->seq++;
	    if (status != 0)
	      {
		m_imap->state = f_imap->state = IMAP_NO_STATE;
		return status;
	      }
	    f_imap->state = IMAP_CREATE;
	  }

	case IMAP_CREATE:
	  status = imap_send (f_imap);
	  if (status != 0)
	    {
	      if (status != EAGAIN && status != EINPROGRESS
		  && status != EINTR)
		m_imap->state = f_imap->state = IMAP_NO_STATE;

	      return status;
	    }
	  f_imap->state = IMAP_CREATE_ACK;

	case IMAP_CREATE_ACK:
	  status = imap_parse (f_imap);
	  if (status != 0)
	    {
	      if (status == EINVAL)
		status = EACCES;

	      if (status != EAGAIN && status != EINPROGRESS
		  && status != EINTR)
		m_imap->state = f_imap->state = IMAP_NO_STATE;

	      return status;
	    }
	  f_imap->state = IMAP_NO_STATE;
	  break;

	default:
	  status = EINVAL;
	  break;
	}
      m_imap->state = IMAP_NO_STATE;
      break;

    default:
      status = EINVAL;
      break;
    }

  return status;
}

/* We can not close the folder in term of shuting down the connection but if
   we were the selected mailbox we send the close and deselect ourself.
   The CLOSE is also use to expunge instead of sending expunge.  */
static int
mailbox_imap_close (mu_mailbox_t mailbox)
{
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;
  
  /* If we are not the selected mailbox, just close the stream.  */
  if (m_imap != f_imap->selected)
    return mu_folder_close (mailbox->folder);

  /* Select first.  */
  status = imap_messages_count (mailbox, NULL);
  if (status != 0)
    return status;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      status = imap_writeline (f_imap, "g%lu CLOSE\r\n",
			       (unsigned long) f_imap->seq++);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_CLOSE;

    case IMAP_CLOSE:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_CLOSE_ACK;

    case IMAP_CLOSE_ACK:
      {
	size_t i;
	status = imap_parse (f_imap);
	CHECK_EAGAIN (f_imap, status);
	MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);

	mu_monitor_wrlock (mailbox->monitor);
	/* Destroy the imap messages and ressources associated to them.  */
	for (i = 0; i < m_imap->imessages_count; i++)
	  {
	    if (m_imap->imessages[i])
	      free_subparts (m_imap->imessages[i]);
	  }
	if (m_imap->imessages)
	  free (m_imap->imessages);
	m_imap->imessages = NULL;
	m_imap->imessages_count = 0;
	m_imap->messages_count = 0;
	m_imap->recent = 0;
	m_imap->unseen = 0;
	/* Clear the callback string structure.  */
	mu_stream_truncate (f_imap->string.stream, 0);
	f_imap->string.offset = 0;
	f_imap->string.nleft = 0;
	f_imap->string.type = IMAP_NO_STATE;
	f_imap->string.msg_imap = NULL;
	mu_monitor_unlock (mailbox->monitor);
      }
      break;

    default:
      /* mu_error ("imap_close unknown state: reconnect\n");*/
      break;
    }

  /* Deselect.  */
  f_imap->selected = NULL;

  f_imap->state = IMAP_NO_STATE;
  return mu_folder_close (mailbox->folder);
}

/* Construction of the mu_message_t, nothing else is done then this setup.  To
   clarify this is different from say mu_message_get_part().  This call is for the
   mailbox and we are setting up the mu_message_t structure.  */
static int
imap_get_message (mu_mailbox_t mailbox, size_t msgno, mu_message_t *pmsg)
{
  m_imap_t m_imap = mailbox->data;
  msg_imap_t msg_imap;
  int status = 0;

  if (pmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (msgno == 0 || msgno > m_imap->messages_count)
    return EINVAL;

  /* Check to see if we have already this message.  */
  mu_monitor_rdlock (mailbox->monitor);
  {
    size_t i;
    for (i = 0; i < m_imap->imessages_count; i++)
      {
	if (m_imap->imessages[i])
	  {
	    if (m_imap->imessages[i]->num == msgno)
	      {
		*pmsg = m_imap->imessages[i]->message;
		mu_monitor_unlock (mailbox->monitor);
		return 0;
	      }
	  }
      }
  }
  mu_monitor_unlock (mailbox->monitor);

  /* Allocate a concrete imap message.  */
  msg_imap = calloc (1, sizeof *msg_imap);
  if (msg_imap == NULL)
    return ENOMEM;
  /* Back pointer.  */
  msg_imap->m_imap = m_imap;
  msg_imap->num = msgno;
  status = imap_get_message0 (msg_imap, pmsg);
  if (status == 0)
    {
      /* Add it to the list.  */
      mu_monitor_wrlock (mailbox->monitor);
      {
	msg_imap_t *m ;
	m = realloc (m_imap->imessages,
		     (m_imap->imessages_count + 1) * sizeof *m);
	if (m == NULL)
	  {
	    mu_message_destroy (pmsg, msg_imap);
	    mu_monitor_unlock (mailbox->monitor);
	    return ENOMEM;
	  }
	m_imap->imessages = m;
	m_imap->imessages[m_imap->imessages_count] = msg_imap;
	m_imap->imessages_count++;
      }
      mu_monitor_unlock (mailbox->monitor);

      msg_imap->message = *pmsg;
    }
  else
    free (msg_imap);
  return status;
}

/* Set all the mu_message_t functions and parts.  */
static int
imap_get_message0 (msg_imap_t msg_imap, mu_message_t *pmsg)
{
  int status = 0;
  mu_message_t msg = NULL;
  mu_mailbox_t mailbox = msg_imap->m_imap->mailbox;

  /* Create the message and its stream.  */
  {
    mu_stream_t stream = NULL;
    if ((status = mu_message_create (&msg, msg_imap)) != 0
        || (status = mu_stream_create (&stream, mailbox->flags, msg)) != 0)
      {
        mu_stream_destroy (&stream, msg);
        mu_message_destroy (&msg, msg_imap);
        return status;
      }
    mu_stream_setbufsiz (stream, 128);
    mu_stream_set_read (stream, imap_message_read, msg);
    mu_stream_set_get_transport2 (stream, imap_message_get_transport2, msg);
    mu_message_set_stream (msg, stream, msg_imap);
    mu_message_set_size (msg, imap_message_size, msg_imap);
    mu_message_set_lines (msg, imap_message_lines, msg_imap);
  }

  /* Create the header.  */
  {
    mu_header_t header = NULL;
    if ((status = mu_header_create (&header, NULL, 0,  msg)) != 0)
      {
        mu_message_destroy (&msg, msg_imap);
        return status;
      }
    mu_header_set_fill (header, imap_header_read, msg);
    mu_message_set_header (msg, header, msg_imap);
  }

  /* Create the attribute.  */
  {
    mu_attribute_t attribute;
    status = mu_attribute_create (&attribute, msg);
    if (status != 0)
      {
        mu_message_destroy (&msg, msg_imap);
        return status;
      }
    mu_attribute_set_get_flags (attribute, imap_attr_get_flags, msg);
    mu_attribute_set_set_flags (attribute, imap_attr_set_flags, msg);
    mu_attribute_set_unset_flags (attribute, imap_attr_unset_flags, msg);
    mu_message_set_attribute (msg, attribute, msg_imap);
  }

  /* Create the body and its stream.  */
  {
    mu_body_t body = NULL;
    mu_stream_t stream = NULL;
    if ((status = mu_body_create (&body, msg)) != 0
        || (status = mu_stream_create (&stream, mailbox->flags, body)) != 0)
      {
        mu_body_destroy (&body, msg);
        mu_stream_destroy (&stream, body);
        mu_message_destroy (&msg, msg_imap);
        return status;
      }
    mu_stream_setbufsiz (stream, 128);
    mu_stream_set_read (stream, imap_body_read, body);
    mu_stream_set_get_transport2 (stream, imap_body_get_transport2, body);
    mu_body_set_size (body, imap_body_size, msg);
    mu_body_set_lines (body, imap_body_lines, msg);
    mu_body_set_stream (body, stream, msg);
    mu_message_set_body (msg, body, msg_imap);
  }

  /* Set the envelope.  */
  {
    mu_envelope_t envelope= NULL;
    status = mu_envelope_create (&envelope, msg);
    if (status != 0)
      {
        mu_message_destroy (&msg, msg_imap);
        return status;
      }
    mu_envelope_set_sender (envelope, imap_envelope_sender, msg);
    mu_envelope_set_date (envelope, imap_envelope_date, msg);
    mu_message_set_envelope (msg, envelope, msg_imap);
  }

  /* Set the mime handling.  */
  mu_message_set_is_multipart (msg, imap_is_multipart, msg_imap);
  mu_message_set_get_num_parts (msg, imap_get_num_parts, msg_imap);
  mu_message_set_get_part (msg, imap_get_part, msg_imap);

  /* Set the UID on the message. */
  mu_message_set_uid (msg, imap_message_uid, msg_imap);
  mu_message_set_mailbox (msg, mailbox, msg_imap);

  /* We are done here.  */
  *pmsg = msg;
  return 0;
}

static int
imap_message_unseen (mu_mailbox_t mailbox, size_t *punseen)
{
  m_imap_t m_imap = mailbox->data;
  *punseen = m_imap->unseen;
  return 0;
}

static int
imap_messages_recent (mu_mailbox_t mailbox, size_t *precent)
{
  m_imap_t m_imap = mailbox->data;
  *precent = m_imap->recent;
  return 0;
}

static int
imap_uidvalidity (mu_mailbox_t mailbox, unsigned long *puidvalidity)
{
  m_imap_t m_imap = mailbox->data;
  *puidvalidity = m_imap->uidvalidity;
  return 0;
}

static int
imap_uidnext (mu_mailbox_t mailbox, size_t *puidnext)
{
  m_imap_t m_imap = mailbox->data;
  *puidnext = m_imap->uidnext;
  return 0;
}

/* There is no explicit call to get the message count.  The count is send on
   a SELECT/EXAMINE command it is also sent async, meaning it will be piggy
   back on other server response as an untag "EXIST" response.  The
   function is also use as a way to select mailbox by other functions.  */
static int
imap_messages_count (mu_mailbox_t mailbox, size_t *pnum)
{
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;

  /* FIXME: It is debatable if we should reconnect when the connection
     timeout or die.  Probably for timeout client should ping i.e. send
     a NOOP via imap_is_updated() function to keep the connection alive.  */
  status = mu_folder_open (mailbox->folder, mailbox->flags);
  if (status != 0)
    return status;

  /* Are we already selected ? */
  if (m_imap == (f_imap->selected))
    {
      if (pnum)
	*pnum = m_imap->messages_count;
      return 0;
    }

  /*  Put the mailbox as selected.  */
  f_imap->selected = m_imap;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      status = imap_writeline (f_imap, "g%lu %s %s\r\n",
			       (unsigned long) f_imap->seq++, 
                               MBX_WRITABLE(mailbox) ? "SELECT" : "EXAMINE",
                               m_imap->name);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_SELECT;

    case IMAP_SELECT:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_SELECT_ACK;

    case IMAP_SELECT_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      break;

    default:
      /*mu_error ("imap_message_count unknown state: reconnect\n");*/
      break;
    }

  if (pnum)
    *pnum = m_imap->messages_count;

  f_imap->state = IMAP_NO_STATE;
  return status;
}

static int
imap_scan (mu_mailbox_t mailbox, size_t msgno, size_t *pcount)
{
  return imap_scan0 (mailbox, msgno, pcount , 1);
}

/* Normally this function is called when an observer is trying to build
   some sort of list/tree header as the scanning progresses.  But doing
   this for each message can be time consuming and inefficient.  So we
   bundle all requests into one and ask the server for everything:
   "FETCH 1:*".  The good side is that everything will be faster and we
   do not do lots of small transcations, but rather a big one.  The bad
   side is that everything will be cached in the structure using a lot of
   memory.  */
static int
imap_scan0 (mu_mailbox_t mailbox, size_t msgno, size_t *pcount, int notif)
{
  int status;
  size_t i;
  size_t count = 0;
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;

  /* Selected.  */
  status = imap_messages_count (mailbox, &count);
  if (pcount)
    *pcount = count;
  if (status != 0)
    return status;

  /* No need to scan, there is no messages. */
  if (count == 0)
    return 0;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      status = imap_writeline (f_imap,
			       "g%lu FETCH 1:* (FLAGS RFC822.SIZE BODY.PEEK[HEADER.FIELDS (%s)])\r\n",
			       (unsigned long) f_imap->seq++,
			       MU_IMAP_CACHE_HEADERS);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_SCAN;

    case IMAP_SCAN:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_SCAN_ACK;
      /* Clear the callback string structure.  */
      mu_stream_truncate (f_imap->string.stream, 0);
      f_imap->string.offset = 0;
      f_imap->string.nleft = 0;
      f_imap->string.type = IMAP_NO_STATE;
      f_imap->string.msg_imap = NULL;

    case IMAP_SCAN_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      /* Clear the callback string structure.  */
      mu_stream_truncate (f_imap->string.stream, 0);
      f_imap->string.offset = 0;
      f_imap->string.nleft = 0;
      f_imap->string.type = IMAP_NO_STATE;
      f_imap->string.msg_imap = NULL;
      break;

    default:
      /*mu_error ("imap_scan unknown state: reconnect\n");*/
      return EINVAL;
    }

  f_imap->state = IMAP_NO_STATE;

  /* Do not send notifications.  */
  if (!notif)
    return 0;

  /* If no callbacks bail out early.  */
  if (mailbox->observable == NULL)
    return 0;

  for (i = msgno; i <= count; i++)
    {
      size_t tmp = i;
      if (mu_observable_notify (mailbox->observable, MU_EVT_MESSAGE_ADD,
                                &tmp) != 0)
	break;
      if ((i + 1) % 100 == 0)
	mu_observable_notify (mailbox->observable, MU_EVT_MAILBOX_PROGRESS, 
                              NULL);
    }
  return 0;
}

/* Send a NOOP and see if the count has changed.  */
static int
imap_is_updated (mu_mailbox_t mailbox)
{
  m_imap_t m_imap = mailbox->data;
  size_t oldcount = m_imap->messages_count;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;

  /* Selected.  */
  status = imap_messages_count (mailbox, &oldcount);
  if (status != 0)
    return status;

  /* Send a noop, and let imap piggy pack the information.  */
  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      status = imap_writeline (f_imap, "g%lu NOOP\r\n",
			       (unsigned long) f_imap->seq++);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_NOOP;

    case IMAP_NOOP:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_NOOP_ACK;

    case IMAP_NOOP_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      break;

    default:
      /*mu_error ("imap_noop unknown state: reconnect\n"); */
      break;
    }
  f_imap->state = IMAP_NO_STATE;
  return (oldcount == m_imap->messages_count);
}


static int
imap_expunge (mu_mailbox_t mailbox)
{
  int status;
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;

  if (!MBX_WRITABLE(mailbox))
    return EACCES;
       
  /* Select first.  */
  status = imap_messages_count (mailbox, NULL);
  if (status != 0)
    return status;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      {
	char *set = NULL;
	status = delete_to_string (m_imap, &set);
	CHECK_ERROR (f_imap, status);
	if (set == NULL || *set == '\0')
	  {
	    if (set)
	      free (set);
	    return 0;
	  }
	status = imap_writeline (f_imap,
				 "g%lu STORE %s +FLAGS.SILENT (\\Deleted)\r\n",
				 (unsigned long) f_imap->seq++,
				 set);
	free (set);
	CHECK_ERROR (f_imap, status);
	MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
	f_imap->state = IMAP_STORE;
      }

      /* Send DELETE.  */
    case IMAP_STORE:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_STORE_ACK;

    case IMAP_STORE_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_NO_STATE;

    case IMAP_EXPUNGE:
    case IMAP_EXPUNGE_ACK:
      status = imap_writeline (f_imap, "g%lu EXPUNGE\r\n",
			       (unsigned long) f_imap->seq++);
      CHECK_ERROR (f_imap, status);
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);

      /* Rescan after expunging but do not trigger the observers.  */
    case IMAP_SCAN:
    case IMAP_SCAN_ACK:
      status = imap_scan0 (mailbox, 1, NULL, 0);
      CHECK_EAGAIN (f_imap, status);

    default:
      /* mu_error ("imap_expunge: unknown state\n"); */
      break;
    }

  return status;
}

/* FIXME: Not ___Nonblocking___ safe.  */
/* DANGER:  The mu_message_t object makes no guaranty about the size and the lines
   that it returns, if its pointing to non-local file messages, so we
   make a local copy.  */
static int
imap_append_message (mu_mailbox_t mailbox, mu_message_t msg)
{
  int status = 0;
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;

  /* FIXME: It is debatable if we should reconnect when the connection
   timeout or die.  For timeout client should ping i.e. send
   a NOOP via imap_is_updated() function to keep the connection alive.  */
  status = mu_folder_open (mailbox->folder, mailbox->flags);
  if (status != 0)
    return status;

  /* FIXME: Can we append to self.  */

  /* Check to see if we are selected. If the message was not modified
     and came from the same imap folder. use COPY.*/
  if (f_imap->selected != m_imap && !mu_message_is_modified (msg)
      && is_same_folder (mailbox, msg))
    return imap_copy_message (mailbox, msg);

  /* copy the message to local disk by createing a floating message.  */
  {
    mu_message_t message = NULL;

    status = mu_message_create_copy(&message, msg);

    if (status == 0)
      status = imap_append_message0 (mailbox, message);
    mu_message_destroy (&message, NULL);
  }
  return status;
}

/* Ok this mean that the message is coming from somewhere else.  IMAP
   is very susceptible on the size, example:
   A003 APPEND saved-messages (\Seen) {310}
   if the server does not get the right size advertise in the string literal
   it will misbehave.  Sine we are assuming that the message will be
   in native file system format meaning ending with NEWLINE, we will have
   to do the calculation.  But what is worse; the value return
   by mu_message_size () and mu_message_lines () are no mean exact but rather
   a gross approximation for certain type of mailbox.  So the sane
   thing to do is to save the message in temporary file, this we say
   we guarantee the size of the message.  */
static int
imap_append_message0 (mu_mailbox_t mailbox, mu_message_t msg)
{
  size_t total;
  int status = 0;
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      {
	size_t lines, size;
	const char *path;
	char *abuf = malloc (1);
	/* Get the desired flags attribute.  */
	if (abuf == NULL)
	  return ENOMEM;
	*abuf = '\0';
	{
	  mu_attribute_t attribute = NULL;
	  int flags = 0;
	  mu_message_get_attribute (msg, &attribute);
	  mu_attribute_get_flags (attribute, &flags);
	  status = flags_to_string (&abuf, flags);
	  if (status != 0)
	    return status;
	  /* Put the surrounding parenthesis, wu-IMAP is sensible to this.  */
	  {
	    char *tmp = calloc (strlen (abuf) + 3, 1);
	    if (tmp == NULL)
	      {
		free (abuf);
		return ENOMEM;
	      }
	    sprintf (tmp, "(%s)", abuf);
	    free (abuf);
	    abuf = tmp;
	  }
	}

	/* Get the mailbox filepath.  */
        status = mu_url_sget_path (mailbox->url, &path);
        if (status == MU_ERR_NOENT)
          path = "INBOX";

	/* FIXME: we need to get the mu_envelope_date and use it.
	   currently it is ignored.  */

	/* Get the total size, assuming that it is in UNIX format.  */
	lines = size = 0;
	mu_message_size (msg, &size);
	mu_message_lines (msg, &lines);
	total = size + lines;
	status = imap_writeline (f_imap, "g%lu APPEND %s %s {%lu}\r\n",
				 (unsigned long) f_imap->seq++,
				 path,
				 abuf,
				 (unsigned long) (size + lines));
	free (abuf);
	CHECK_ERROR (f_imap, status);
	MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
	f_imap->state = IMAP_APPEND;
      }

    case IMAP_APPEND:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_APPEND_CONT;

    case IMAP_APPEND_CONT:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      /* If we did not receive the continuation token, it is an error
         bail out.  */
      if (f_imap->buffer[0] != '+')
	{
	  status = EACCES;
	  break;
	}
      f_imap->state = IMAP_APPEND_SEND;

    case IMAP_APPEND_SEND:
      {
	mu_stream_t stream = NULL;
	mu_off_t off = 0;
	size_t n = 0;
	char buffer[255];
	mu_message_get_stream (msg, &stream);
	while (mu_stream_readline (stream, buffer, sizeof buffer, off, &n) == 0
	       && n > 0)
	  {
	    if (buffer[n - 1] == '\n')
	      {
		buffer[n - 1] = '\0';
		status = imap_writeline (f_imap, "%s\r\n", buffer);
	      }
	    else
	      imap_writeline (f_imap, "%s", buffer);
	    off += n;
	    status = imap_send (f_imap);
	    CHECK_EAGAIN (f_imap, status);
	  }
	f_imap->state = IMAP_APPEND_ACK;
      }
      /* !@#%$ UW-IMAP and Gimap server hack: both insist on the last line.  */
      imap_writeline (f_imap, "\r\n");
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);

    case IMAP_APPEND_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);

    default:
      /* mu_error ("imap_append: unknown state\n"); */
      break;
    }
  f_imap->state = IMAP_NO_STATE;
  return status;
}

/* If the message is on the same server.  Use the COPY command much more
   efficient.  */
static int
imap_copy_message (mu_mailbox_t mailbox, mu_message_t msg)
{
  m_imap_t m_imap = mailbox->data;
  f_imap_t f_imap = m_imap->f_imap;
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  int status = 0;

  /* FIXME: It is debatable if we should reconnect when the connection
   timeout or die.  For timeout client should ping i.e. send
   a NOOP via imap_is_updated() function to keep the connection alive.  */
  status = mu_folder_open (mailbox->folder, mailbox->flags);
  if (status != 0)
    return status;

  switch (f_imap->state)
    {
    case IMAP_NO_STATE:
      {
	const char *path;
	/* Check for a valid mailbox name.  */
	status = mu_url_sget_path (mailbox->url, &path);
	if (status == 0)
  	  status = imap_writeline (f_imap, "g%lu COPY %lu %s\r\n",
	 			   (unsigned long) f_imap->seq++,
				   (unsigned long) msg_imap->num,
				   path);
	CHECK_ERROR (f_imap, status);
	MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
	f_imap->state = IMAP_COPY;
      }

    case IMAP_COPY:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      f_imap->state = IMAP_COPY_ACK;

    case IMAP_COPY_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      MU_DEBUG (mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);

    default:
      break;
    }
  f_imap->state = IMAP_NO_STATE;
  return status;
}

/* Message read overload  */
static int
imap_message_read (mu_stream_t stream, char *buffer, size_t buflen,
		   mu_off_t offset, size_t *plen)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  char *oldbuf = NULL;
  char newbuf[2];
  int status;

  /* This is so annoying, a buffer len of 1 is a killer. If you have for
     example "\n" to retrieve from the server, IMAP will transform this to
     "\r\n" and since you ask for only 1, the server will send '\r' only.
     And ... '\r' will be stripped by (imap_readline()) the number of char
     read will be 0 which means we're done .... sigh ...  So we guard by at
     least ask for 2 chars.  */
  if (buflen == 1)
    {
      oldbuf = buffer;
      buffer = newbuf;
      buflen = 2;
    }

  /* Start over.  */
  if (offset == 0)
    msg_imap->mu_message_lines = 0;

  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  /* Select first.  */
  if (f_imap->state == IMAP_NO_STATE)
    {
      char *section = NULL;

      if (msg_imap->part)
	section = section_name (msg_imap);

      /* We have strip the \r, but the offset on the imap server is with that
	 octet(CFLF) so add it in the offset, it's the number of lines.  */
      status = imap_writeline (f_imap,
			       "g%lu FETCH %lu BODY.PEEK[%s]<%lu.%lu>\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num,
			       (section) ? section : "",
			       (unsigned long) (offset +
						msg_imap->mu_message_lines),
			       (unsigned long) buflen);
      if (section)
	free (section);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;
    }
  status = fetch_operation (f_imap, msg_imap, buffer, buflen, plen);

  if (oldbuf)
    oldbuf[0] = buffer[0];
  return status;
}

static int
imap_message_lines (mu_message_t msg, size_t *plines)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  if (plines && msg_imap)
    {
      if (msg_imap->mu_message_lines == 0)
	*plines = msg_imap->body_lines + msg_imap->header_lines;
      else
	*plines = msg_imap->mu_message_lines;
    }
  return 0;
}

/* Sometimes a message is just a place container for other sub parts.
   In those cases imap bodystructure does not set the mu_message_size aka
   the mu_body_size.  But we can calculate it since the mu_message_size
   is the sum of its subparts.  */
static int
imap_submessage_size (msg_imap_t msg_imap, size_t *psize)
{
  if (psize)
    {
      *psize = 0;
      if (msg_imap->mu_message_size == 0)
	{
	  size_t i, size;
	  for (size = i = 0; i < msg_imap->num_parts; i++, size = 0)
	    {
	      if (msg_imap->parts[i])
		imap_submessage_size (msg_imap->parts[i], &size);
	      *psize += size;
	    }
	}
      else
	*psize = (msg_imap->mu_message_size + msg_imap->header_size)
	  - msg_imap->mu_message_lines;
    }
  return 0;
}

static int
imap_message_size (mu_message_t msg, size_t *psize)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;;

  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  /* If there is a parent it means it is a sub message, IMAP does not give
     the full size of mime messages, so the mu_message_size retrieved from
     doing a bodystructure represents rather the mu_body_size.  */
  if (msg_imap->parent)
    return imap_submessage_size (msg_imap, psize);

  if (msg_imap->mu_message_size == 0)
    {
      /* Select first.  */
      if (f_imap->state == IMAP_NO_STATE)
	{
	  /* We strip the \r, but the offset/size on the imap server is with
	     that octet so add it in the offset, since it's the number of
	     lines.  */
	  status = imap_writeline (f_imap,
				   "g%lu FETCH %lu RFC822.SIZE\r\n",
				   (unsigned long) f_imap->seq++,
				   (unsigned long) msg_imap->num);
	  CHECK_ERROR (f_imap, status);
	  MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
	  f_imap->state = IMAP_FETCH;
	}
      status = fetch_operation (f_imap, msg_imap, 0, 0, 0);
    }

  if (status == 0)
    {
      if (psize)
	*psize = msg_imap->mu_message_size - msg_imap->mu_message_lines;
    }
  return status;
}

static int
imap_message_uid (mu_message_t msg, size_t *puid)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status;

  if (puid)
    return 0;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  if (f_imap->state == IMAP_NO_STATE)
    {
      if (msg_imap->uid)
	{
	  *puid = msg_imap->uid;
	  return 0;
	}
      status = imap_writeline (f_imap, "g%lu FETCH %lu UID\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;
    }
  status = fetch_operation (f_imap, msg_imap, 0, 0, 0);
  if (status != 0)
    return status;
  *puid = msg_imap->uid;
  return 0;
}

static int
imap_message_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  mu_message_t msg = mu_stream_get_owner (stream);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  return imap_get_transport2 (msg_imap, pin, pout);
}

/* Mime.  */
static int
imap_is_multipart (mu_message_t msg, int *ismulti)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  if (f_imap->state == IMAP_NO_STATE)
    {
      if (msg_imap->num_parts || msg_imap->part)
	{
	  if (ismulti)
	    *ismulti = (msg_imap->num_parts > 1);
	  return 0;
	}
      status = imap_writeline (f_imap,
			       "g%lu FETCH %lu BODYSTRUCTURE\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;
    }
  status = fetch_operation (f_imap, msg_imap, 0, 0, 0);
  if (status != 0)
    return status;
  if (ismulti)
    *ismulti = (msg_imap->num_parts > 1);
  return 0;
}

static int
imap_get_num_parts (mu_message_t msg, size_t *nparts)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  if (msg_imap)
    {
      if (msg_imap->num_parts == 0)
	{
	  int status = imap_is_multipart (msg, NULL);
	  if (status != 0)
	    return status;
	}
      if (nparts)
	*nparts = (msg_imap->num_parts == 0) ? 1 : msg_imap->num_parts;
    }
  return 0;
}

static int
imap_get_part (mu_message_t msg, size_t partno, mu_message_t *pmsg)
{
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  int status = 0;

  if (msg_imap->num_parts == 0)
    {
      status = imap_get_num_parts (msg, NULL);
      if (status != 0)
	return status;
    }

  if (partno <= msg_imap->num_parts)
    {
      if (msg_imap->parts[partno - 1]->message)
	{
	  if (pmsg)
	    *pmsg = msg_imap->parts[partno - 1]->message;
	}
      else
	{
	  mu_message_t message;
	  status = imap_get_message0 (msg_imap->parts[partno - 1], &message);
	  if (status == 0)
	    {
	      mu_header_t header;
	      mu_message_get_header (message, &header);
	      mu_message_set_stream (message, NULL, msg_imap->parts[partno - 1]);
	      /* mu_message_set_size (message, NULL, msg_imap->parts[partno - 1]); */
	      msg_imap->parts[partno - 1]->message = message;
	      if (pmsg)
		*pmsg = message;
	    }
	}
    }
  else
    {
      if (pmsg)
	*pmsg = msg_imap->message;
    }
  return status;
}

/* Envelope.  */
static int
imap_envelope_sender (mu_envelope_t envelope, char *buffer, size_t buflen,
		      size_t *plen)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  mu_header_t header;
  const char *sender;
  int status;

  mu_message_get_header (msg, &header);
  status = mu_header_sget_value (header, MU_HEADER_SENDER, &sender);
  if (status == EAGAIN)
    return status;
  else if (status != 0)
    status = mu_header_sget_value (header, MU_HEADER_FROM, &sender);
  if (status == 0)
    {
      const char *email = NULL;
      size_t len;
      mu_address_t address;
      if (mu_address_create (&address, sender) == 0)
	{
	  if (mu_address_sget_email (address, 1, &email) == 0)
	    len = mu_cpystr (buffer, email, buflen);
	  mu_address_destroy (&address);
	}

      if (!email)
	return MU_ERR_NOENT;
      
      if (plen)
	*plen = len;
    }
  return status;
}

static int
imap_envelope_date (mu_envelope_t envelope, char *buffer, size_t buflen,
		    size_t *plen)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  struct tm tm;
  mu_timezone tz;
  time_t now;
  char datebuf[] = "mm-dd-yyyy hh:mm:ss +0000";
  const char* date = datebuf;
  const char** datep = &date;
  /* reserve as much space as we need for internal-date */
  int status;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;
  if (msg_imap->internal_date == NULL)
    {
      if (f_imap->state == IMAP_NO_STATE)
	{
	  status = imap_writeline (f_imap,
				   "g%lu FETCH %lu INTERNALDATE\r\n",
				   (unsigned long) f_imap->seq++,
				   (unsigned long) msg_imap->num);
	  CHECK_ERROR (f_imap, status);
	  MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
	  f_imap->state = IMAP_FETCH;
	}
      status = fetch_operation (f_imap, msg_imap, datebuf,
				sizeof datebuf, NULL);
      if (status != 0)
	return status;
      msg_imap->internal_date = strdup (datebuf);
    }
  else
    {
      date = msg_imap->internal_date;
      datep = &date;
    }

  if (mu_parse_imap_date_time(datep, &tm, &tz) != 0)
    now = (time_t)-1;
  else
    now = mu_tm2time (&tm, &tz);

  /* if the time was unparseable, or mktime() didn't like what we
     parsed, use the calendar time. */
  if (now == (time_t)-1)
    {
      struct tm *gmt;

      time (&now);
      gmt = gmtime (&now);
      tm = *gmt;
    }

  {
    char tmpbuf[MU_ENVELOPE_DATE_LENGTH+1];
    size_t n = mu_strftime (tmpbuf, sizeof tmpbuf,
                            MU_ENVELOPE_DATE_FORMAT, &tm);
    n = mu_cpystr (buffer, tmpbuf, buflen);
    if (plen)
      *plen = n;
  }
  return 0;
}

/* Attributes.  */
static int
imap_attr_get_flags (mu_attribute_t attribute, int *pflags)
{
  mu_message_t msg = mu_attribute_get_owner (attribute);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  /* Did we retrieve it alread ?  */
  if (msg_imap->flags != 0)
    {
      if (pflags)
	*pflags = msg_imap->flags;
      return 0;
    }

  if (f_imap->state == IMAP_NO_STATE)
    {
      status = imap_writeline (f_imap, "g%lu FETCH %lu FLAGS\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;
    }
  status = fetch_operation (f_imap, msg_imap, NULL, 0, NULL);
  if (status == 0)
    {
      if (pflags)
	*pflags = msg_imap->flags;
    }
  return status;
}

static int
imap_attr_set_flags (mu_attribute_t attribute, int flag)
{
  mu_message_t msg = mu_attribute_get_owner (attribute);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  /* If already set don't bother.  */
  if (msg_imap->flags & flag)
    return 0;

  /* The delete FLAG is not pass yet but only on the expunge.  */
  if (flag & MU_ATTRIBUTE_DELETED)
    {
      msg_imap->flags |= MU_ATTRIBUTE_DELETED;
      flag &= ~MU_ATTRIBUTE_DELETED;
    }

  if (f_imap->state == IMAP_NO_STATE)
    {
      char *abuf = malloc (1);
      if (abuf == NULL)
	return ENOMEM;
      *abuf = '\0';
      status = flags_to_string (&abuf, flag);
      if (status != 0)
	return status;
      /* No flags to send??  */
      if (*abuf == '\0')
	{
	  free (abuf);
	  return 0;
	}
      status = imap_writeline (f_imap, "g%lu STORE %lu +FLAGS.SILENT (%s)\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num,
			       abuf);
      free (abuf);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      msg_imap->flags |= flag;
      f_imap->state = IMAP_FETCH;
    }
  return fetch_operation (f_imap, msg_imap, NULL, 0, NULL);
}

static int
imap_attr_unset_flags (mu_attribute_t attribute, int flag)
{
  mu_message_t msg = mu_attribute_get_owner (attribute);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  int status = 0;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  /* The delete FLAG is not pass yet but only on the expunge.  */
  if (flag & MU_ATTRIBUTE_DELETED)
    {
      msg_imap->flags &= ~MU_ATTRIBUTE_DELETED;
      flag &= ~MU_ATTRIBUTE_DELETED;
    }

  if (f_imap->state == IMAP_NO_STATE)
    {
      char *abuf = malloc (1);
      if (abuf == NULL)
	return ENOMEM;
      *abuf = '\0';
      status = flags_to_string (&abuf, flag);
      if (status != 0)
	return status;
      /* No flags to send??  */
      if (*abuf == '\0')
	{
	  free (abuf);
	  return 0;
	}
      status = imap_writeline (f_imap, "g%lu STORE %lu -FLAGS.SILENT (%s)\r\n",
			       (unsigned long) f_imap->seq++,
			       (unsigned long) msg_imap->num,
			       abuf);
      free (abuf);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      msg_imap->flags &= ~flag;
      f_imap->state = IMAP_FETCH;
    }
  return fetch_operation (f_imap, msg_imap, NULL, 0, NULL);
}

/* Header.  */
static int
imap_header_read (mu_header_t header, char *buffer,
		  size_t buflen, mu_off_t offset,
		  size_t *plen)
{
  mu_message_t msg = mu_header_get_owner (header);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  char *oldbuf = NULL;
  char newbuf[2];
  int status;

  /* This is so annoying, a buffer len of 1 is a killer. If you have for
     example "\n" to retrieve from the server, IMAP will transform this to
     "\r\n" and since you ask for only 1, the server will send '\r' only.
     And ... '\r' will be stripped by (imap_readline()) the number of char
     read will be 0 which means we're done .... sigh ...  So we guard by at
     least ask for 2 chars.  */
  if (buflen == 1)
    {
      oldbuf = buffer;
      buffer = newbuf;
      buflen = 2;
    }

  /* Start over.  */
  if (offset == 0)
    msg_imap->header_lines = 0;

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  if (f_imap->state == IMAP_NO_STATE)
    {
      /* We strip the \r, but the offset/size on the imap server is with that
         octet so add it in the offset, since it's the number of lines.  */
      if (msg_imap->part)
        {
          char *section = section_name (msg_imap);
          status = imap_writeline (f_imap,
                                   "g%lu FETCH %lu BODY.PEEK[%s.MIME]<%lu.%lu>\r\n",
                                   (unsigned long) f_imap->seq++,
				   (unsigned long) msg_imap->num,
                                   (section) ? section : "",
                                   (unsigned long) (offset +
						    msg_imap->header_lines),
				   (unsigned long) buflen);
          if (section)
            free (section);
        }
      else
        status = imap_writeline (f_imap,
                                 "g%lu FETCH %lu BODY.PEEK[HEADER]<%lu.%lu>\r\n",
                                 (unsigned long) f_imap->seq++,
				 (unsigned long) msg_imap->num,
                                 (unsigned long) (offset +
						  msg_imap->header_lines),
				 (unsigned long) buflen);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;

    }
  status = fetch_operation (f_imap, msg_imap, buffer, buflen, plen);
  if (oldbuf)
    oldbuf[0] = buffer[0];
  return status;
}

/* Body.  */
static int
imap_body_size (mu_body_t body, size_t *psize)
{
  mu_message_t msg = mu_body_get_owner (body);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  if (psize && msg_imap)
    {
      /* If there is a parent it means it is a sub message, IMAP does not give
	 the full size of mime messages, so the mu_message_size was retrieve from
	 doing a bodystructure and represents rather the mu_body_size.  */
      if (msg_imap->parent)
	{
	  *psize = msg_imap->mu_message_size - msg_imap->mu_message_lines;
	}
      else
	{
	  if (msg_imap->body_size)
	    *psize = msg_imap->body_size;
	  else if (msg_imap->mu_message_size)
	    *psize = msg_imap->mu_message_size
	      - (msg_imap->header_size + msg_imap->header_lines);
	  else
	    *psize = 0;
	}
    }
  return 0;
}

static int
imap_body_lines (mu_body_t body, size_t *plines)
{
  mu_message_t msg = mu_body_get_owner (body);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  if (plines && msg_imap)
    *plines = msg_imap->body_lines;
  return 0;
}

/* FIXME: Send EISPIPE if trying to seek back.  */
static int
imap_body_read (mu_stream_t stream, char *buffer, size_t buflen,
		mu_off_t offset, size_t *plen)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  m_imap_t m_imap = msg_imap->m_imap;
  f_imap_t f_imap = m_imap->f_imap;
  char *oldbuf = NULL;
  char newbuf[2];
  int status;

  /* This is so annoying, a buffer len of 1 is a killer. If you have for
     example "\n" to retrieve from the server, IMAP will transform this to
     "\r\n" and since you ask for only 1, the server will send '\r' only.
     And ... '\r' will be stripped by (imap_readline()) the number of char
     read will be 0 which means we're done .... sigh ...  So we guard by at
     least ask for 2 chars.  */
  if (buflen == 1)
    {
      oldbuf = buffer;
      buffer = newbuf;
      buflen = 2;
    }

  /* Start over.  */
  if (offset == 0)
    {
      msg_imap->body_lines = 0;
      msg_imap->body_size = 0;
    }

  /* Select first.  */
  status = imap_messages_count (m_imap->mailbox, NULL);
  if (status != 0)
    return status;

  if (f_imap->state == IMAP_NO_STATE)
    {
      /* We strip the \r, but the offset/size on the imap server is with the
         octet, so add it since it's the number of lines.  */
      if (msg_imap->part)
        {
          char *section = section_name (msg_imap);
          status = imap_writeline (f_imap,
                                   "g%lu FETCH %lu BODY.PEEK[%s]<%lu.%lu>\r\n",
                                   (unsigned long) f_imap->seq++,
				   (unsigned long) msg_imap->num,
                                   (section) ? section: "",
                                   (unsigned long) (offset +
						    msg_imap->body_lines),
				   (unsigned long) buflen);
          if (section)
            free (section);
        }
      else
        status = imap_writeline (f_imap,
                                 "g%lu FETCH %lu BODY.PEEK[TEXT]<%lu.%lu>\r\n",
                                 (unsigned long) f_imap->seq++,
				 (unsigned long) msg_imap->num,
                                 (unsigned long) (offset +
						  msg_imap->body_lines),
				 (unsigned long) buflen);
      CHECK_ERROR (f_imap, status);
      MU_DEBUG (m_imap->mailbox->debug, MU_DEBUG_PROT, f_imap->buffer);
      f_imap->state = IMAP_FETCH;

    }
  status = fetch_operation (f_imap, msg_imap, buffer, buflen, plen);
  if (oldbuf)
    oldbuf[0] = buffer[0];
  return status;
}

static int
imap_body_get_transport2 (mu_stream_t stream, mu_transport_t *pin, 
                         mu_transport_t *pout)
{
  mu_body_t body = mu_stream_get_owner (stream);
  mu_message_t msg = mu_body_get_owner (body);
  msg_imap_t msg_imap = mu_message_get_owner (msg);
  return imap_get_transport2 (msg_imap, pin, pout);
}


static int
imap_get_transport2 (msg_imap_t msg_imap, mu_transport_t *pin, mu_transport_t *pout)
{
  if (   msg_imap
      && msg_imap->m_imap
      && msg_imap->m_imap->f_imap
      && msg_imap->m_imap->f_imap->folder)
    return mu_stream_get_transport2 (msg_imap->m_imap->f_imap->folder->stream,
			         pin, pout);
  return EINVAL;
}

/* Since so many operations are fetch, we regoup this into one function.  */
static int
fetch_operation (f_imap_t f_imap, msg_imap_t msg_imap, char *buffer,
		 size_t buflen, size_t *plen)
{
  int status = 0;

  switch (f_imap->state)
    {
    case IMAP_FETCH:
      status = imap_send (f_imap);
      CHECK_EAGAIN (f_imap, status);
      mu_stream_truncate (f_imap->string.stream, 0);
      f_imap->string.offset = 0;
      f_imap->string.nleft = 0;
      f_imap->string.type = IMAP_NO_STATE;
      f_imap->string.msg_imap = msg_imap;
      f_imap->state = IMAP_FETCH_ACK;

    case IMAP_FETCH_ACK:
      status = imap_parse (f_imap);
      CHECK_EAGAIN (f_imap, status);
      if (f_imap->selected)
	MU_DEBUG (f_imap->selected->mailbox->debug, MU_DEBUG_PROT,
	   	  f_imap->buffer);

    default:
      break;
    }

  f_imap->state = IMAP_NO_STATE;

  /* The server may have timeout any case connection is gone away.  */
  if (status == 0 && f_imap->isopen == 0 && f_imap->string.offset == 0)
    status = MU_ERR_CONN_CLOSED;

  if (buffer)
    mu_stream_read (f_imap->string.stream, buffer, buflen, 0, plen);
  else if (plen)
    *plen = 0;
  mu_stream_truncate (f_imap->string.stream, 0);
  f_imap->string.offset = 0;
  f_imap->string.nleft = 0;
  f_imap->string.type = IMAP_NO_STATE;
  f_imap->string.msg_imap = NULL;
  return status;
}

/* Decide whether the message came from the same folder as the mailbox.  */
static int
is_same_folder (mu_mailbox_t mailbox, mu_message_t msg)
{
  mu_mailbox_t mbox = NULL;
  mu_message_get_mailbox (msg, &mbox);
  return (mbox != NULL && mbox->url != NULL
          && mu_url_is_same_scheme (mbox->url, mailbox->url)
          && mu_url_is_same_host (mbox->url, mailbox->url)
          && mu_url_is_same_port (mbox->url, mailbox->url));
}

/* Convert flag attribute to IMAP String attributes.  */
static int
flags_to_string (char **pbuf, int flag)
{
  char *abuf = *pbuf;
  if (flag & MU_ATTRIBUTE_DELETED)
    {
      char *tmp = realloc (abuf, strlen (abuf) + strlen ("\\Deleted") + 2);
      if (tmp == NULL)
        {
          free (abuf);
          return ENOMEM;
        }
      abuf = tmp;
      if (*abuf)
        strcat (abuf, " ");
      strcat (abuf, "\\Deleted");
    }
  if (flag & MU_ATTRIBUTE_READ)
    {
      char *tmp = realloc (abuf, strlen (abuf) + strlen ("\\Seen") + 2);
      if (tmp == NULL)
        {
          free (abuf);
          return ENOMEM;
        }
      abuf = tmp;
      if (*abuf)
        strcat (abuf, " ");
      strcat (abuf, "\\Seen");
    }
  if (flag & MU_ATTRIBUTE_ANSWERED)
    {
      char *tmp = realloc (abuf, strlen (abuf) + strlen ("\\Answered") + 2);
      if (tmp == NULL)
        {
          free (abuf);
          return ENOMEM;
        }
      abuf = tmp;
      if (*abuf)
        strcat (abuf, " ");
      strcat (abuf, "\\Answered");
    }
  if (flag & MU_ATTRIBUTE_DRAFT)
    {
      char *tmp = realloc (abuf, strlen (abuf) + strlen ("\\Draft") + 2);
      if (tmp == NULL)
        {
          free (abuf);
          return ENOMEM;
        }
      abuf = tmp;
      if (*abuf)
        strcat (abuf, " ");
      strcat (abuf, "\\Draft");
    }
  if (flag & MU_ATTRIBUTE_FLAGGED)
    {
      char *tmp = realloc (abuf, strlen (abuf) + strlen ("\\Flagged") + 2);
      if (tmp == NULL)
        {
          free (abuf);
          return ENOMEM;
        }
      abuf = tmp;
      if (*abuf)
        strcat (abuf, " ");
      strcat (abuf, "\\Flagged");
    }
  *pbuf = abuf;
  return 0;
}

/* Convert a suite of number to IMAP message number.  */
static int
add_number (char **pset, size_t start, size_t end)
{
  char *buf = NULL;
  char *set;
  char *tmp;
  size_t set_len = 0;

  if (pset == NULL)
    return 0;

  set = *pset;

  if (set)
    set_len = strlen (set);

  /* We had a previous seqence.  */
  if (start == 0)
    /* nothing */;
  else if (start != end)
    mu_asprintf (&buf, "%lu:%lu",
		  (unsigned long) start, (unsigned long) end);
  else
    mu_asprintf (&buf, "%lu", (unsigned long) start);

  if (set_len)
    tmp = realloc (set, set_len + strlen (buf) + 2 /* null and comma */);
  else
    tmp = calloc (strlen (buf) + 1, 1);

  if (tmp == NULL)
    {
      free (set);
      free (buf);
      return ENOMEM;
    }
  set = tmp;

  /* If we had something add a comma separator.  */
  if (set_len)
    strcat (set, ",");
  strcat (set, buf);
  free (buf);
  
  *pset = set;
  return 0;
}

static int
delete_to_string (m_imap_t m_imap, char **pset)
{
  int status;
  size_t i, prev = 0, is_range = 0;
  size_t start = 0, cur = 0;
  char *set = NULL;

  /* Reformat the number for IMAP.  */
  for (i = 0; i < m_imap->imessages_count; ++i)
    {
      if (m_imap->imessages[i]
	  && (m_imap->imessages[i]->flags & MU_ATTRIBUTE_DELETED))
	{
	  cur = m_imap->imessages[i]->num;
	  /* The first number.  */
	  if (start == 0)
	    {
	      start = prev = cur;
	    }
	  /* Is it a sequence?  */
	  else if ((prev + 1) == cur)
	    {
	      prev = cur;
	      is_range = 1;
	    }
	  continue;
	}

      if (start)
	{
	  status = add_number (&set, start, cur);
	  if (status != 0)
	    return status;
	  start = 0;
	  prev = 0;
	  cur = 0;
	  is_range = 0;
	}
    } /* for () */

  if (start)
    {
      status = add_number (&set, start, cur);
      if (status != 0)
	return status;
    }
  *pset = set;
  return 0;
}

#endif
