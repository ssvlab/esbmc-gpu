/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2008, 2009,
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
#include <config.h>
#endif

#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <mailutils/debug.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/folder.h>
#include <mailutils/iterator.h>
#include <mailutils/list.h>
#include <mailutils/locker.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/registrar.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/attribute.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>

#include <mailbox0.h>
#include <url0.h>

static int
mailbox_folder_create (mu_mailbox_t mbox, const char *name,
		       mu_record_t record)
{
  int rc;
  mu_url_t url;
  
  if ((rc = mu_url_uplevel (mbox->url, &url)))
    {
      if (rc == MU_ERR_NOENT)
	{
	  rc = mu_url_dup (mbox->url, &url);
	  if (rc)
	    return rc;
	}
      else
	return rc;
    }

  rc = mu_folder_create_from_record (&mbox->folder, url, record);
  if (rc)
    mu_url_destroy (&url);
  return rc;
}

static int
_create_mailbox0 (mu_mailbox_t *pmbox, mu_url_t url, const char *name)
{
  int status;
  mu_record_t record = NULL;

  if (mu_registrar_lookup_url (url, MU_FOLDER_ATTRIBUTE_FILE, &record, NULL)
      == 0)
    {
      mu_log_level_t level;
      int (*m_init) (mu_mailbox_t) = NULL;
      
      mu_record_get_mailbox (record, &m_init);
      if (m_init)
        {
	  int (*u_init) (mu_url_t) = NULL;
	  mu_mailbox_t mbox;

	  /* Allocate memory for mbox.  */
	  mbox = calloc (1, sizeof (*mbox));
	  if (mbox == NULL)
	    return ENOMEM;

	  /* Initialize the internal lock now, so the concrete mailbox
	     could use it. */
	  status = mu_monitor_create (&mbox->monitor, 0, mbox);
	  if (status != 0)
	    {
	      mu_mailbox_destroy (&mbox);
	      return status;
	    }

	  /* Make sure scheme contains actual mailbox scheme */
	  /* FIXME: It is appropriate not for all record types.  For now we
	     assume that if the record scheme ends with a plus sign, this
	     should not be done.  Probably it requires some flag in struct
	     _mu_record? */
	  if (strcmp (url->scheme, record->scheme))
	    {
	      char *p = strdup (record->scheme);
	      if (!p)
		{
		  mu_mailbox_destroy (&mbox);
		  return errno;
		}
	      free (url->scheme);
	      url->scheme = p;
	    }

	  mu_record_get_url (record, &u_init);
	  if (u_init && (status = u_init (url)) != 0)
	    {
	      mu_mailbox_destroy (&mbox);
	      return status;
	    }

	  mbox->url = url;

	  /* Create the folder before initializing the concrete mailbox.
	     The mailbox needs it's back pointer. */
	  status = mailbox_folder_create (mbox, name, record);
	  
	  if (status == 0)
	    status = m_init (mbox);   /* Create the concrete mailbox type.  */

	  if (status != 0)
	    {
	      /* Take care not to destroy url.  Leave it to caller. */
	      mbox->url = NULL;
	      mu_mailbox_destroy (&mbox);
	    }
	  else
	    {
	      *pmbox = mbox;

	      level = mu_global_debug_level ("mailbox");
	      if (level)
		{
		  int status = mu_debug_create (&mbox->debug, mbox);
		  if (status)
		    return 0; /* FIXME: don't want to bail out just because I
				 failed to create a *debug* object. But I may
				 be wrong... */
		  mu_debug_set_level (mbox->debug, level);
		  if (level & MU_DEBUG_INHERIT)
		    mu_folder_set_debug (mbox->folder, mbox->debug);
		}
	    }
	  
	  return status;
	}
    }
  return MU_ERR_NO_HANDLER;
}

static int
_create_mailbox (mu_mailbox_t *pmbox, const char *name)
{
  int status;
  mu_url_t url;

  status = mu_url_create (&url, name);
  if (status)
    return status;
  status = mu_url_parse (url);
  if (status == 0)
    status = _create_mailbox0 (pmbox, url, name);
  if (status)
    mu_url_destroy (&url);
  return status;
}

/* The Mailbox Factory.
   Create an iterator for registrar and see if any url scheme match,
   Then we call the mailbox's mu_url_create() to parse the URL. Last
   initialize the concrete mailbox and folder.  */
int
mu_mailbox_create (mu_mailbox_t *pmbox, const char *name)
{
  if (pmbox == NULL)
    return MU_ERR_OUT_PTR_NULL;

  return _create_mailbox (pmbox, name);
}

int
mu_mailbox_create_from_url (mu_mailbox_t *pmbox, mu_url_t url)
{
  if (pmbox == NULL)
    return MU_ERR_OUT_PTR_NULL;
  return _create_mailbox0 (pmbox, url, mu_url_to_string (url));
}

void
mu_mailbox_destroy (mu_mailbox_t *pmbox)
{
  if (pmbox && *pmbox)
    {
      mu_mailbox_t mbox = *pmbox;
      mu_monitor_t monitor = mbox->monitor;

      /* Notify the observers.  */
      if (mbox->observable)
	{
	  mu_observable_notify (mbox->observable, MU_EVT_MAILBOX_DESTROY,
				mbox);
	  mu_observable_destroy (&mbox->observable, mbox);
	}

      /* Call the concrete mailbox _destroy method. So it can clean itself.  */
      if (mbox->_destroy)
	mbox->_destroy (mbox);

      mu_monitor_wrlock (monitor);

      /* Close the stream and nuke it */
      if (mbox->stream)
	{
	  /* FIXME: Is this right, should the client be responsible
	     for closing the stream?  */
	  /* mu_stream_close (mbox->stream); */
	  mu_stream_destroy (&mbox->stream, mbox);
	}

      if (mbox->url)
        mu_url_destroy (&mbox->url);

      if (mbox->locker)
	mu_locker_destroy (&mbox->locker);

      if (mbox->debug)
	mu_debug_destroy (&mbox->debug, mbox);

      if (mbox->folder)
	mu_folder_destroy (&mbox->folder);

      if (mbox->property)
	mu_property_destroy (&mbox->property, mbox);

      free (mbox);
      *pmbox = NULL;
      mu_monitor_unlock (monitor);
      mu_monitor_destroy (&monitor, mbox);
    }
}


/* -------------- stub functions ------------------- */

int
mu_mailbox_open (mu_mailbox_t mbox, int flag)
{
  if (mbox == NULL || mbox->_open == NULL)
    return MU_ERR_EMPTY_VFN;
  if (flag & MU_STREAM_QACCESS)
    {
      /* Quick access mailboxes are read-only */
      if (flag & (MU_STREAM_WRITE | MU_STREAM_RDWR
		  | MU_STREAM_APPEND | MU_STREAM_CREAT))
	return EINVAL; /* FIXME: Better error code, please? */
    }
  return mbox->_open (mbox, flag);
}

int
mu_mailbox_close (mu_mailbox_t mbox)
{
  if (mbox == NULL || mbox->_close == NULL)
    return MU_ERR_EMPTY_VFN;

  return mbox->_close (mbox);
}

int
mu_mailbox_flush (mu_mailbox_t mbox, int expunge)
{
  size_t i, total = 0;
  int status = 0;
  
  if (!mbox)
    return EINVAL;
  if (!(mbox->flags & (MU_STREAM_RDWR|MU_STREAM_WRITE|MU_STREAM_APPEND)))
    return 0;

  mu_mailbox_messages_count (mbox, &total);
  if (!(mbox->flags & MU_STREAM_APPEND))
    for (i = 1; i <= total; i++)
      {
	mu_message_t msg = NULL;
	mu_attribute_t attr = NULL;
	mu_mailbox_get_message (mbox, i, &msg);
	mu_message_get_attribute (msg, &attr);
	mu_attribute_set_seen (attr);
      }

  if (expunge)
    status = mu_mailbox_expunge (mbox);
  else
    status = mu_mailbox_sync (mbox);

  return status;
}

/* messages */
int
mu_mailbox_append_message (mu_mailbox_t mbox, mu_message_t msg)
{
  if (mbox == NULL || mbox->_append_message == NULL)
    return MU_ERR_EMPTY_VFN;
  if (!(mbox->flags & (MU_STREAM_RDWR|MU_STREAM_WRITE|MU_STREAM_APPEND)))
    return EACCES;
  return mbox->_append_message (mbox, msg);
}

int
mu_mailbox_get_message (mu_mailbox_t mbox, size_t msgno,  mu_message_t *pmsg)
{
  if (mbox == NULL || mbox->_get_message == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_get_message (mbox, msgno, pmsg);
}

int
mu_mailbox_quick_get_message (mu_mailbox_t mbox, mu_message_qid_t qid,
			      mu_message_t *pmsg)
{
  if (mbox == NULL || mbox->_quick_get_message == NULL)
    return MU_ERR_EMPTY_VFN;
  if (!(mbox->flags & MU_STREAM_QACCESS))
    return MU_ERR_BADOP;
  return mbox->_quick_get_message (mbox, qid, pmsg);
}

int
mu_mailbox_messages_count (mu_mailbox_t mbox, size_t *num)
{
  if (mbox == NULL || mbox->_messages_count == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_messages_count (mbox, num);
}

int
mu_mailbox_messages_recent (mu_mailbox_t mbox, size_t *num)
{
  if (mbox == NULL || mbox->_messages_recent == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_messages_recent (mbox, num);
}

int
mu_mailbox_message_unseen (mu_mailbox_t mbox, size_t *num)
{
  if (mbox == NULL || mbox->_message_unseen == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_message_unseen (mbox, num);
}

int
mu_mailbox_sync (mu_mailbox_t mbox)
{
  if (mbox == NULL || mbox->_sync == NULL)
    return MU_ERR_EMPTY_VFN;
  if (!(mbox->flags & (MU_STREAM_RDWR|MU_STREAM_WRITE|MU_STREAM_APPEND)))
    return 0;
  return mbox->_sync (mbox);
}

/* Historic alias: */
int
mu_mailbox_save_attributes (mu_mailbox_t mbox)
{
  if (mbox == NULL || mbox->_sync == NULL)
    return MU_ERR_EMPTY_VFN;
  if (!(mbox->flags & (MU_STREAM_RDWR|MU_STREAM_WRITE|MU_STREAM_APPEND)))
    return EACCES;
  return mbox->_sync (mbox);
}

int
mu_mailbox_expunge (mu_mailbox_t mbox)
{
  if (mbox == NULL || mbox->_expunge == NULL)
    return MU_ERR_EMPTY_VFN;
  if (!(mbox->flags & (MU_STREAM_RDWR|MU_STREAM_WRITE|MU_STREAM_APPEND)))
    return EACCES;
  return mbox->_expunge (mbox);
}

int
mu_mailbox_is_updated (mu_mailbox_t mbox)
{
  if (mbox == NULL || mbox->_is_updated == NULL)
    return 1;
  if (mbox->flags & MU_STREAM_QACCESS)
    return 1;
  return mbox->_is_updated (mbox);
}

int
mu_mailbox_scan (mu_mailbox_t mbox, size_t msgno, size_t *pcount)
{
  if (mbox == NULL || mbox->_scan == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_scan (mbox, msgno, pcount);
}

int
mu_mailbox_get_size (mu_mailbox_t mbox, mu_off_t *psize)
{
  int status;
  if (mbox == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  if (mbox->_get_size == NULL
      || (status = mbox->_get_size (mbox, psize)) == ENOSYS)
    {
      /* Fall back to brute-force method */
      size_t i, total;
      mu_off_t size = 0;
      
      status = mu_mailbox_messages_count (mbox, &total);
      if (status)
	return status;
      for (i = 1; i <= total; i++)
	{
	  mu_message_t msg;
	  size_t msgsize;
	  status = mu_mailbox_get_message (mbox, i, &msg);
	  if (status)
	    return status;
	  status = mu_message_size (msg, &msgsize);
	  if (status)
	    return status;
	  size += msgsize;
	}
      *psize = size;
    }
  return status;
}

int
mu_mailbox_uidvalidity (mu_mailbox_t mbox, unsigned long *pvalid)
{
  if (mbox == NULL || mbox->_uidvalidity == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_uidvalidity (mbox, pvalid);
}

int
mu_mailbox_uidnext (mu_mailbox_t mbox, size_t *puidnext)
{
  if (mbox == NULL || mbox->_uidnext == NULL)
    return MU_ERR_EMPTY_VFN;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  return mbox->_uidnext (mbox, puidnext);
}

/* locking */
int
mu_mailbox_set_locker (mu_mailbox_t mbox, mu_locker_t locker)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (mbox->locker)
    mu_locker_destroy (&mbox->locker);
  mbox->locker = locker;
  return 0;
}

int
mu_mailbox_get_locker (mu_mailbox_t mbox, mu_locker_t *plocker)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (plocker == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *plocker = mbox->locker;
  return 0;
}

int
mu_mailbox_get_flags (mu_mailbox_t mbox, int *flags)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (!*flags)
    return MU_ERR_OUT_NULL;
  *flags = mbox->flags;
  return 0;
}

int
mu_mailbox_set_stream (mu_mailbox_t mbox, mu_stream_t stream)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (mbox->flags & MU_STREAM_QACCESS)
    return MU_ERR_BADOP;
  if (mbox->stream)
    mu_stream_destroy (&mbox->stream, mbox);
  mbox->stream = stream;
  return 0;
}

/* FIXME: This is a problem.  We provide a mu_mailbox_get_stream ()
   and this stream is special it should, in theory, represent
   a "view" of a flow of messages.  But providing this perspective
   may make sense for local mailboxes but downright impossible
   for a remote mailbox, short on downloading the entire mailbox
   locally.
   The question is : should this function be removed?
   So far it as been used on local mailboxes to get offsets.  */
int
mu_mailbox_get_stream (mu_mailbox_t mbox, mu_stream_t *pstream)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (pstream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  /* If null two cases:
     - it is no open yet.
     - it a remote stream and the socket stream is on the folder.  */
  if (mbox->stream == NULL)
    {
      if (mbox->folder)
	return mu_folder_get_stream (mbox->folder, pstream);
    }
  *pstream = mbox->stream;
  return 0;
}

int
mu_mailbox_get_observable (mu_mailbox_t mbox, mu_observable_t *pobservable)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (pobservable == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (mbox->observable == NULL)
    {
      int status = mu_observable_create (&mbox->observable, mbox);
      if (status != 0)
	return status;
    }
  *pobservable = mbox->observable;
  return 0;
}

int
mu_mailbox_get_property (mu_mailbox_t mbox, mu_property_t *pproperty)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (pproperty == NULL)
    return MU_ERR_OUT_PTR_NULL;
  
  if (mbox->property == NULL)
    {
      int status = mu_property_create (&mbox->property, mbox);
      if (status != 0)
	return status;
    }
  *pproperty = mbox->property;
  return 0;
}

int
mu_mailbox_has_debug (mu_mailbox_t mailbox)
{
  if (mailbox == NULL)
    return 0;

  return mailbox->debug ? 1 : 0;
}

int
mu_mailbox_set_debug (mu_mailbox_t mbox, mu_debug_t debug)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (mbox->debug)
    mu_debug_destroy (&mbox->debug, mbox);
  mbox->debug = debug;
  /* FIXME: Honor MU_DEBUG_INHERIT */
  if (!mu_folder_has_debug (mbox->folder))
    mu_folder_set_debug (mbox->folder, debug);
  return 0;
}

int
mu_mailbox_get_debug (mu_mailbox_t mbox, mu_debug_t *pdebug)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (pdebug == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (mbox->debug == NULL)
    {
      int status = mu_debug_create (&mbox->debug, mbox);
      if (status != 0)
	return status;
      /* FIXME: MU_DEBUG_INHERIT?? */
      if (!mu_folder_has_debug (mbox->folder))
	mu_folder_set_debug (mbox->folder, mbox->debug);
    }
  *pdebug = mbox->debug;
  return 0;
}

int
mu_mailbox_get_url (mu_mailbox_t mbox, mu_url_t *purl)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (purl == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *purl = mbox->url;
  return 0;
}

int
mu_mailbox_get_folder (mu_mailbox_t mbox, mu_folder_t *pfolder)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (pfolder == NULL)
    return MU_ERR_OUT_PTR_NULL;
  *pfolder = mbox->folder;
  return 0;
}

int
mu_mailbox_set_folder (mu_mailbox_t mbox, mu_folder_t folder)
{
  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
   mbox->folder = folder;
  return 0;
}

int
mu_mailbox_lock (mu_mailbox_t mbox)
{
  mu_locker_t lock = NULL;
  mu_mailbox_get_locker (mbox, &lock);
  return mu_locker_lock (lock);
}

int
mu_mailbox_unlock (mu_mailbox_t mbox)
{
  mu_locker_t lock = NULL;
  mu_mailbox_get_locker (mbox, &lock);
  return mu_locker_unlock (lock);
}

int
mu_mailbox_get_uidls (mu_mailbox_t mbox, mu_list_t *plist)
{
  mu_list_t list;
  int status;

  if (mbox == NULL)
    return MU_ERR_MBX_NULL;
  if (plist == NULL)
    return EINVAL;
  status = mu_list_create (&list);
  if (status)
    return status;
  mu_list_set_destroy_item (list, mu_list_free_item);
  if (mbox->_get_uidls)
    status = mbox->_get_uidls (mbox, list);
  else
    {
      size_t i, total;

      status = mu_mailbox_messages_count (mbox, &total);
      if (status)
	return status;
      for (i = 1; i <= total; i++)
	{
	  mu_message_t msg = NULL;
	  char buf[MU_UIDL_BUFFER_SIZE];
	  size_t n;
	  struct mu_uidl *uidl;
	  
	  status = mu_mailbox_get_message (mbox, i, &msg);
	  if (status)
	    break;
	  status = mu_message_get_uidl (msg, buf, sizeof (buf), &n);
	  if (status)
	    break;
	  uidl = malloc (sizeof (uidl[0]));
	  if (!uidl)
	    {
	      status = ENOMEM;
	      break;
	    }
	  uidl->msgno = i;
	  strncpy (uidl->uidl, strdup (buf), MU_UIDL_BUFFER_SIZE);
	  status = mu_list_append (list, uidl);
	  if (status)
	    {
	      free (uidl);
	      break;
	    }
	}
    }
  *plist = list;
  return status;
}

	  
