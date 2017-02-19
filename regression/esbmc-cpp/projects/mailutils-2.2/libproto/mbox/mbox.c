/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

/* First draft by Alain Magloire */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <mbox0.h>
#include <mailutils/cstr.h>
#include <mailutils/io.h>

#define ATTRIBUTE_IS_DELETED(flag)        (flag & MU_ATTRIBUTE_DELETED)
#define ATTRIBUTE_IS_EQUAL(flag1, flag2)  (flag1 == flag2)

static void mbox_destroy (mu_mailbox_t);

/* Mailbox concrete implementation.  */
static int mbox_open                  (mu_mailbox_t, int);
static int mbox_close                 (mu_mailbox_t);
static int mbox_get_message           (mu_mailbox_t, size_t, mu_message_t *);
static int mbox_quick_get_message (mu_mailbox_t, mu_message_qid_t,
				   mu_message_t *);

/* static int mbox_get_message_by_uid    (mu_mailbox_t, size_t, mu_message_t *); */
static int mbox_append_message        (mu_mailbox_t, mu_message_t);
static int mbox_messages_count        (mu_mailbox_t, size_t *);
static int mbox_messages_recent       (mu_mailbox_t, size_t *);
static int mbox_message_unseen        (mu_mailbox_t, size_t *);
static int mbox_expunge0              (mu_mailbox_t, int);
static int mbox_expunge               (mu_mailbox_t);
static int mbox_sync                  (mu_mailbox_t);
static int mbox_uidvalidity           (mu_mailbox_t, unsigned long *);
static int mbox_uidnext               (mu_mailbox_t, size_t *);
static int mbox_scan                  (mu_mailbox_t, size_t, size_t *);
static int mbox_is_updated            (mu_mailbox_t);
static int mbox_get_size              (mu_mailbox_t, mu_off_t *);

/* private stuff */
static int mbox_append_message0       (mu_mailbox_t, mu_message_t,
				       mu_off_t *, int, int);
static int mbox_message_uid           (mu_message_t, size_t *);
static int mbox_message_qid           (mu_message_t, mu_message_qid_t *);

static int mbox_header_fill           (mu_header_t, char *, size_t,
				       mu_off_t, size_t *);
static int mbox_get_body_transport    (mu_stream_t, mu_transport_t *,
				       mu_transport_t *);
static int mbox_get_transport2        (mbox_message_t, mu_transport_t *,
				       mu_transport_t *);
static int mbox_get_attr_flags        (mu_attribute_t, int *);
static int mbox_set_attr_flags        (mu_attribute_t, int);
static int mbox_unset_attr_flags      (mu_attribute_t, int);
static int mbox_body_read             (mu_stream_t, char *, size_t,
				       mu_off_t, size_t *);
static int mbox_body_readline         (mu_stream_t, char *, size_t,
				       mu_off_t, size_t *);
static int mbox_readstream            (mbox_message_t, char *, size_t,
				       mu_off_t, size_t *, int, mu_off_t,
				       mu_off_t);
static int mbox_stream_size           (mu_stream_t stream, mu_off_t *psize);

static int mbox_body_size             (mu_body_t, size_t *);
static int mbox_body_lines            (mu_body_t, size_t *);
static int mbox_envelope_sender       (mu_envelope_t, char *, size_t,
				       size_t *);
static int mbox_envelope_date         (mu_envelope_t, char *, size_t,
				       size_t *);
static int mbox_tmpfile               (mu_mailbox_t, char **pbox);

/* Allocate the mbox_data_t struct(concrete mailbox), but don't do any
   parsing on the name or even test for existence.  However we do strip any
   leading "mbox:" part of the name, this is suppose to be the
   protocol/scheme name.  */
int
_mailbox_mbox_init (mu_mailbox_t mailbox)
{
  int status;
  mbox_data_t mud;

  if (mailbox == NULL)
    return EINVAL;

  /* Allocate specific mbox data.  */
  mud = mailbox->data = calloc (1, sizeof (*mud));
  if (mailbox->data == NULL)
    return ENOMEM;

  /* Back pointer.  */
  mud->mailbox = mailbox;

  /* Copy the name:
     We do not do any further interpretation after the scheme "mbox:"
     Because for example on distributed system like QnX4 a file name is
     //390/etc/passwd.  So the best approach is to let the OS handle it
     for example if we receive: "mbox:///var/mail/alain" the mailbox name
     will be "///var/mail/alain", we let open() do the right thing.
     So it will let things like this "mbox://390/var/mail/alain" where
     the "//" _is_ part of the filename, pass correctely.  */
  status = mu_url_aget_path (mailbox->url, &mud->name);
  if (status)
    {
      free (mud);
      mailbox->data = NULL;
      return status;
    }

  mud->state = MBOX_NO_STATE;

  /* Overloading the defaults.  */
  mailbox->_destroy = mbox_destroy;

  mailbox->_open = mbox_open;
  mailbox->_close = mbox_close;

  /* Overloading of the entire mailbox object methods.  */
  mailbox->_get_message = mbox_get_message;
  mailbox->_append_message = mbox_append_message;
  mailbox->_messages_count = mbox_messages_count;
  mailbox->_messages_recent = mbox_messages_recent;
  mailbox->_message_unseen = mbox_message_unseen;
  mailbox->_expunge = mbox_expunge;
  mailbox->_sync = mbox_sync;
  mailbox->_uidvalidity = mbox_uidvalidity;
  mailbox->_uidnext = mbox_uidnext;
  mailbox->_quick_get_message = mbox_quick_get_message;

  mailbox->_scan = mbox_scan;
  mailbox->_is_updated = mbox_is_updated;

  mailbox->_get_size = mbox_get_size;

  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailbox_get_property (mailbox, &property);
    mu_property_set_value (property, "TYPE", "MBOX", 1);
  }

  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_init (%s)\n", mud->name);
  return 0; /* okdoke */
}

/* Free all ressources associated with Unix concrete mailbox.  */
static void
mbox_destroy (mu_mailbox_t mailbox)
{
  if (mailbox->data)
    {
      size_t i;
      mbox_data_t mud = mailbox->data;
      MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1,
		      "mbox_destroy (%s)\n", mud->name);
      mu_monitor_wrlock (mailbox->monitor);
      for (i = 0; i < mud->umessages_count; i++)
	{
	  mbox_message_t mum = mud->umessages[i];
	  if (mum)
	    {
	      mu_message_destroy (&(mum->message), mum);
	      free (mum);
	    }
	}
      if (mud->umessages)
	free (mud->umessages);
      if (mud->name)
	free (mud->name);
      free (mud);
      mailbox->data = NULL;
      mu_monitor_unlock (mailbox->monitor);
    }
}

/* Open the file.  For MU_STREAM_READ, the code tries mmap() first and fall
   back to normal file.  */
static int
mbox_open (mu_mailbox_t mailbox, int flags)
{
  mbox_data_t mud = mailbox->data;
  int status = 0;

  if (mud == NULL)
    return EINVAL;

  mailbox->flags = flags;

  /* Get a stream.  */
  if (mailbox->stream == NULL)
    {
      /* We do not try to mmap for CREAT or APPEND, it is not supported.  */
      status = (flags & MU_STREAM_CREAT)
	|| (mailbox->flags & MU_STREAM_APPEND);

      /* Try to mmap () the file first.  */
      if (status == 0)
	{
	  status = mu_mapfile_stream_create (&mailbox->stream, mud->name, mailbox->flags);
	  if (status == 0)
	    {
	      status = mu_stream_open (mailbox->stream);
	    }
	}

      /* Fall back to normal file if mmap() failed.  */
      if (status != 0)
	{
	  status = mu_file_stream_create (&mailbox->stream, mud->name, mailbox->flags);
	  if (status != 0)
	    return status;
	  status = mu_stream_open (mailbox->stream);
	}
      /* All failed, bail out.  */
      if (status != 0)
	{
	  mu_stream_destroy (&mailbox->stream, NULL);
	  return status;
	}
      /* Even on top of normal FILE *, lets agressively cache.  But this
	 may not be suitable for system tight on memory.  */
      mu_stream_setbufsiz (mailbox->stream, BUFSIZ);
    }
  else
    {
      status = mu_stream_open (mailbox->stream);
      if (status != 0)
	return status;
    }

  MU_DEBUG2 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_open (%s, 0x%x)\n",
	     mud->name, mailbox->flags);

  if (mailbox->locker == NULL)
    status = mu_locker_create (&(mailbox->locker), mud->name, 0);
  return status;
}

static int
mbox_close (mu_mailbox_t mailbox)
{
  mbox_data_t mud = mailbox->data;
  size_t i; 

  if (mud == NULL)
    return EINVAL;

  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1,  "mbox_close (%s)\n", mud->name);

  /* Make sure that we do not hold any file locking.  */
  mu_locker_unlock (mailbox->locker);

  /* Alain: I'm not sure on the right approach especially if the client is
     working in disconnected mode, where it can mu_mailbox_close/mu_mailbox_open
     for each request, maybe we should keep them for a while.

     Sergey: No, it actually breaks reopening the mailbox. We should make
     sure that the sequence mu_mailbox_open();mu_mailbox_close() will catch all
     the changes that might have been done to the mailbox */
  
  mu_monitor_wrlock (mailbox->monitor);
  /* Before closing we need to remove all the messages
     - to reclaim the memory
     - to prepare for another scan.  */
  for (i = 0; i < mud->umessages_count; i++)
    {
      mbox_message_t mum = mud->umessages[i];
      /* Destroy the attach messages.  */
      if (mum)
	{
	  mu_message_destroy (&(mum->message), mum);
	  free (mum);
	}
    }
  if (mud->umessages)
    free (mud->umessages);
  mud->umessages = NULL;
  mud->messages_count = mud->umessages_count = 0;
  mud->size = 0;
  mud->uidvalidity = 0;
  mud->uidnext = 0;
  mu_monitor_unlock (mailbox->monitor);

  return mu_stream_close (mailbox->stream);
}

/* Cover function that call the real thing, mbox_scan(), with
   notification set.  */
static int
mbox_scan (mu_mailbox_t mailbox, size_t msgno, size_t *pcount)
{
  size_t i;
  mbox_data_t mud = mailbox->data;
  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_scan (%s)\n", mud->name);
  if (! mbox_is_updated (mailbox))
    return mbox_scan0 (mailbox, msgno, pcount, 1);
  /* Since the mailbox is already updated fake the scan. */
  if (msgno > 0)
    msgno--; /* The fist message is number "1", decrement for the C array.  */
  for (i = msgno; i < mud->messages_count; i++)
    {
      size_t tmp = i;
      if (mu_observable_notify (mailbox->observable, MU_EVT_MESSAGE_ADD,
				&tmp) != 0)
	break;
      if (((i +1) % 50) == 0)
	{
	  mu_observable_notify (mailbox->observable, MU_EVT_MAILBOX_PROGRESS,
				NULL);
	}
    }
  *pcount = mud->messages_count;
  return 0;
}

/* Alain: How to handle a shrink ? meaning, the &^$^@%#@^& user start two
   browsers and deleted emails in one session.  My views is that we should
   scream bloody murder and hunt them with a machette. But for now just play
   dumb, but maybe the best approach is to pack our things and leave
   .i.e exit()/abort(). 

   Sergey: Nope, we shouldn't abort. Handling it with MU_EVT_MAILBOX_CORRUPT
   is sensible enough. The caller must decide what's the best approach
   in this case. The simplest one is reopening the mailbox. Imap4d currently
   does that. */

static int
mbox_is_updated (mu_mailbox_t mailbox)
{
  mu_off_t size = 0;
  mbox_data_t mud = mailbox->data;
  if (mu_stream_size (mailbox->stream, &size) != 0)
    return 1;
  if (size < mud->size)
    {
      mu_observable_notify (mailbox->observable, MU_EVT_MAILBOX_CORRUPT,
			    mailbox);
      /* And be verbose.  ? */
      mu_diag_output (MU_DIAG_EMERG, _("mailbox corrupted, shrank in size"));
      /* FIXME: should I crash.  */
      return 0;
    }
  return (mud->size == size);
}

/* Try to create an uniq file, we no race conditions.   */
static int
mbox_tmpfile (mu_mailbox_t mailbox, char **pbox)
{
  const char *tmpdir;
  int fd;
  const char *basename;
  mbox_data_t mud = mailbox->data;

  /*  P_tmpdir should be in <stdio.h>.  */
#ifndef P_tmpdir
#  define P_tmpdir "/tmp"
#endif

  basename = strrchr (mud->name, '/');
  if (basename)
    basename++;
  else
    basename = mud->name;

  tmpdir =  getenv ("TMPDIR") ? getenv ("TMPDIR") : P_tmpdir;
  /* (separator + null) == 2 + XXXXXX == 6 + ... */
  *pbox = calloc (strlen (tmpdir) + /* '/' */ 1 + /*strlen ("MU_")*/ 3 +
		  strlen (basename) + /*strlen ("_XXXXXX")*/ 7 + /*null*/1,
		  sizeof (**pbox));
  if (*pbox == NULL)
    return -1;
  sprintf (*pbox, "%s/MU_%s_XXXXXX", tmpdir, basename);
#ifdef HAVE_MKSTEMP
  fd = mkstemp (*pbox);
#else
  /* Create the file.  It must not exist.  If it does exist, fail.  */
  if (mktemp (*pbox))
    {
      fd = open (*pbox, O_RDWR|O_CREAT|O_EXCL, 0600);
    }
  else
    {
      free (*pbox);
      fd = -1;
    }
#endif
  return fd;
}

/* For the expunge bits  we took a very cautionnary approach, meaning
   we create a temporary mailbox in the tmpdir copy all the message not mark
   deleted(Actually we copy all the message that may have been modified
   i.e new header values set; UIDL or UID or etc ....
   and skip the deleted ones, truncate the real mailbox to the desired size
   and overwrite with the tmp mailbox.  The approach to do everyting
   in core is tempting but require
   - to much memory, it is not rare nowadays to have 30 Megs mailbox,
   - also there is danger for filesystems with quotas,
   - if we run out of disk space everything is lost.
   - or some program may not respect the advisory lock and decide to append
   a new message while your expunging etc ...
   The real downside to the approach is that when things go wrong
   the temporary file may be left in /tmp, which is not all that bad
   because at least, we have something to recuperate when failure.  */
static int
mbox_expunge0 (mu_mailbox_t mailbox, int remove_deleted)
{
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum;
  int status = 0;
  sigset_t signalset;
  int tempfile;
  size_t i, j, dirty;  /* dirty will indicate the first modified message.  */
  mu_off_t marker = 0;    /* marker will be the position to truncate.  */
  mu_off_t total = 0;
  char *tmpmboxname = NULL;
  mu_mailbox_t tmpmailbox = NULL;
  size_t save_imapbase = 0;  /* uidvalidity is save in the first message.  */
#ifdef WITH_PTHREAD
  int state;
#endif

  if (mud == NULL)
    return EINVAL;

  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_expunge (%s)\n", mud->name);

  /* Noop.  */
  if (mud->messages_count == 0)
    return 0;

  /* Find the first dirty(modified) message.  */
  for (dirty = 0; dirty < mud->messages_count; dirty++)
    {
      mum = mud->umessages[dirty];
      /* Message may have been tampered, break here.  */
      if ((mum->attr_flags & MU_ATTRIBUTE_MODIFIED) ||
	  (mum->attr_flags & MU_ATTRIBUTE_DELETED) ||
	  (mum->message && mu_message_is_modified (mum->message)))
	break;
    }

  /* Did something change ?  */
  if (dirty == mud->messages_count)
    return 0; /* Nothing change, bail out.  */

  /* Create a temporary file.  */
  tempfile = mbox_tmpfile (mailbox, &tmpmboxname);
  if (tempfile == -1)
    {
      if (tmpmboxname)
	free (tmpmboxname);
      mu_error (_("failed to create temporary file when expunging"));
      return errno;
    }

  /* This is redundant, we go to the loop again.  But it's more secure here
     since we don't want to be disturb when expunging.  Destroy all the
     messages mark for deletion.  */
  if (remove_deleted)
    {
      for (j = 0; j < mud->messages_count; j++)
	{
	  mum = mud->umessages[j];
	  if (mum && mum->message && ATTRIBUTE_IS_DELETED (mum->attr_flags))
	    mu_message_destroy (&(mum->message), mum);
	}
    }

  /* Create temporary mu_mailbox_t.  */
  {
    mbox_data_t tmp_mud;
    char *m = malloc (5 + strlen (tmpmboxname) + 1);
    if (!m)
      return ENOMEM;
    /* Try via the mbox: protocol.  */
    sprintf (m, "mbox:%s", tmpmboxname);
    status = mu_mailbox_create (&tmpmailbox, m);
    if (status != 0)
      {
	/* Do not give up just yet, maybe they register the mu_path_record.  */
	sprintf (m, "%s", tmpmboxname);
	status = mu_mailbox_create (&tmpmailbox, m);
	if (status != 0)
	  {
	    /* Ok give up.  */
	    close (tempfile);
	    remove (tmpmboxname);
            free (m);
	    free (tmpmboxname);
	    return status;
	  }
      }
    free (m);
    /* Must be flag CREATE if not the mu_mailbox_open will try to mmap()
       the file.  */
    status = mu_mailbox_open (tmpmailbox, MU_STREAM_CREAT | MU_STREAM_RDWR);
    if (status != 0)
      {
	close (tempfile);
	remove (tmpmboxname);
	free (tmpmboxname);
	return status;
      }
    close (tempfile); /* This one is useless the mailbox have its own.  */
    tmp_mud = tmpmailbox->data;
    /* May need when appending.  */
    tmp_mud->uidvalidity = mud->uidvalidity;
    tmp_mud->uidnext = mud->uidnext;
  }

  /* Get the File lock.  */
  if ((status = mu_locker_lock (mailbox->locker)) != 0)
    {
      mu_mailbox_close (tmpmailbox);
      mu_mailbox_destroy (&tmpmailbox);
      remove (tmpmboxname);
      free (tmpmboxname);
      mu_error (_("failed to grab the lock: %s"), mu_strerror (status));
      return status;
    }

  /* Critical section, we can not allowed signal here.  */
#ifdef WITH_PTHREAD
  pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &state);
#endif
  sigemptyset (&signalset);
  sigaddset (&signalset, SIGTERM);
  sigaddset (&signalset, SIGHUP);
  sigaddset (&signalset, SIGTSTP);
  sigaddset (&signalset, SIGINT);
  sigaddset (&signalset, SIGWINCH);
  sigprocmask (SIG_BLOCK, &signalset, 0);

  /* Set the marker position.  */
  marker = mud->umessages[dirty]->header_from;
  total = 0;

  /* Copy to the temporary mailbox emails not mark deleted.  */
  for (i = dirty; i < mud->messages_count; i++)
    {
      mum = mud->umessages[i];

      /* Skip it, if mark for deletion.  */
      if (remove_deleted && ATTRIBUTE_IS_DELETED (mum->attr_flags))
	{
	  /* We save the uidvalidity in the first message, if it is being
	     deleted we need to move the uidvalidity to the first available
	     (non-deleted) message.  */
	  if (i == save_imapbase)
	    {
	      save_imapbase = i + 1;
	      if (save_imapbase < mud->messages_count)
		(mud->umessages[save_imapbase])->attr_flags |= MU_ATTRIBUTE_MODIFIED;
	    }
	  continue;
	}

      /* Do the expensive mbox_append_message0() only if mark dirty.  */
      if ((mum->attr_flags & MU_ATTRIBUTE_MODIFIED) ||
	  (mum->message && mu_message_is_modified (mum->message)))
	{
	  /* The message was not instantiated, probably the dirty flag was
	     set by mbox_scan(), create one here.  */
	  if (mum->message == 0)
	    {
	      mu_message_t msg;
	      status = mbox_get_message (mailbox, i + 1, &msg);
	      if (status != 0)
		{
		  mu_error (_("error expunging:%d: %s"), __LINE__,
			    mu_strerror (status));
		  goto bailout0;
		}
	    }
	  status = mbox_append_message0 (tmpmailbox, mum->message,
					 &total, 1, (i == save_imapbase));
	  if (status != 0)
	    {
	      mu_error (_("error expunging:%d: %s"), __LINE__,
		        mu_strerror (status));
	      goto bailout0;
	    }
	  /* Clear the dirty bits.  */
	  mum->attr_flags &= ~MU_ATTRIBUTE_MODIFIED;
	  mu_message_clear_modified (mum->message);
	}
      else
	{
	  /* Nothing changed copy the message straight.  */
	  char buffer [1024];
	  size_t n;
	  mu_off_t offset = mum->header_from;
	  size_t len = mum->body_end - mum->header_from;
	  while (len > 0)
	    {
	      n = (len < sizeof (buffer)) ? len : sizeof (buffer);
	      if ((status = mu_stream_read (mailbox->stream, buffer, n, offset,
					 &n) != 0)
		  || (status = mu_stream_write (tmpmailbox->stream, buffer, n,
					     total, &n) != 0))
		{
		  mu_error (_("error expunging:%d: %s"), __LINE__,
			    mu_strerror (status));
		  goto bailout0;
		}
	      len -= n;
	      total += n;
	      offset += n;
	    }
	  /* Add the newline separator.  */
	  status = mu_stream_write (tmpmailbox->stream, "\n", 1, total, &n);
	  if (status != 0)
	    {
	      mu_error (_("error expunging:%d: %s"), __LINE__,
		        mu_strerror (status));
	      goto bailout0;
	    }
	  total++;
	}
    } /* for (;;) */

  /* Caution: before ftruncate()ing the file see
     - if we've receive new mails.  Some programs may not respect the lock,
     - or the lock was held for too long.
     - The mailbox may not have been properly updated before expunging.  */
  {
    mu_off_t size = 0;
    if (mu_stream_size (mailbox->stream, &size) == 0)
      {
	mu_off_t len = size - mud->size;
	mu_off_t offset = mud->size;
	char buffer [1024];
	size_t n = 0;
	if (len > 0 )
	  {
	    while ((status = mu_stream_read (mailbox->stream, buffer,
					  sizeof (buffer), offset, &n)) == 0
		   && n > 0)
	      {
		status = mu_stream_write (tmpmailbox->stream, buffer, n,
				       total, &n);
		if (status != 0)
		  {
		    mu_error (_("error expunging:%d: %s"), __LINE__,
			      mu_strerror (status));
		    goto bailout0;
		  }
		total += n;
		offset += n;
	      }
	  }
	else if (len < 0)
	  {
	    /* Corrupted mailbox.  */
	    mu_error (_("error expunging:%d: %s"), __LINE__,
		      mu_strerror (status));
	    goto bailout0;
	  }
      }
  } /* End of precaution.  */

  /* Seek and rewrite it.  */
  if (total > 0)
    {
      char buffer [1024];
      size_t n = 0;
      mu_off_t off = 0;
      mu_off_t offset = marker;
      while ((status = mu_stream_read (tmpmailbox->stream, buffer,
				    sizeof (buffer), off, &n)) == 0
	     && n > 0)
	{
	  status = mu_stream_write (mailbox->stream, buffer, n, offset, &n);
	  if (status != 0)
	    {
	      mu_error (_("error expunging:%d: %s"), __LINE__,
		        mu_strerror (status));
	      goto bailout;
	    }
	  off += n;
	  offset += n;
	}
    }

  /* Flush/truncation. Need to flush before truncate.  */
  mu_stream_flush (mailbox->stream);
  status = mu_stream_truncate (mailbox->stream, total + marker);
  if (status != 0)
    {
      mu_error (_("error expunging:%d: %s"), __LINE__,
	        mu_strerror (status));
      goto bailout;
    }

  /* Don't remove the tmp mbox in case of errors, when writing back.  */
 bailout0:
  remove (tmpmboxname);

 bailout:

  free (tmpmboxname);
  /* Release the File lock.  */
  mu_locker_unlock (mailbox->locker);
  mu_mailbox_close (tmpmailbox);
  mu_mailbox_destroy (&tmpmailbox);

  /* Reenable signal.  */
#ifdef WITH_PTHREAD
  pthread_setcancelstate (state, &state);
#endif
  sigprocmask (SIG_UNBLOCK, &signalset, 0);

  /* We need to readjust the pointers.
     It is a little hairy because we need to keep the message pointers alive
     So we are going through the array and "defragmentize".  For example
     in (1 2 3 4) if 2 was deleted we need to move 3 4 up by one etc ..
  */
  if (status == 0)
    {
      size_t dlast;
      mu_monitor_wrlock (mailbox->monitor);
      for (j = dirty, dlast = mud->messages_count - 1;
	   j <= dlast; j++)
	{
	  /* Clear all the references, any attach messages been already
	     destroy above.  */
	  mum = mud->umessages[j];
	  if (remove_deleted && ATTRIBUTE_IS_DELETED (mum->attr_flags))
	    {
	      if ((j + 1) <= dlast)
		{
		  /* Move all the pointers up.  So the message pointer
		     part of mum will be at the right position.  */
		  memmove (mud->umessages + j, mud->umessages + j + 1,
			   (dlast - j) * sizeof (mum));
#if 0
		  mum->header_from = mum->header_from_end = 0;
		  mum->body = mum->body_end = 0;
		  mum->header_lines = mum->body_lines = 0;
#endif
		  memset (mum, 0, sizeof (*mum));
		  /* We are not free()ing the useless mum, but instead
		     we put it back in the pool, to be reuse.  */
		  mud->umessages[dlast] = mum;
		  dlast--;
		  /* Set mum to the new value after the memmove so it
		     gets cleared to.  */
		  mum = mud->umessages[j];
		}
	      else
		{
		  memset (mum, 0, sizeof (*mum));
		}
	    }
	  mum->header_from = mum->header_from_end = 0;
	  mum->body = mum->body_end = 0;
	  mum->header_lines = mum->body_lines = 0;
	}
      mu_monitor_unlock (mailbox->monitor);
      /* This is should reset the messages_count, the last argument 0 means
	 not to send event notification.  */
      mbox_scan0 (mailbox, dirty, NULL, 0);
    }
  return status;
}

static int
mbox_expunge (mu_mailbox_t mailbox)
{
  return mbox_expunge0 (mailbox, 1);
}

static int
mbox_sync (mu_mailbox_t mailbox)
{
  return mbox_expunge0 (mailbox, 0);
}

static int
mbox_message_uid (mu_message_t msg, size_t *puid)
{
  mbox_message_t mum = mu_message_get_owner (msg);
  if (puid)
    *puid = mum->uid;
  return 0;
}

static int
mbox_message_qid (mu_message_t msg, mu_message_qid_t *pqid)
{
  mbox_message_t mum = mu_message_get_owner (msg);
  return mu_asprintf (pqid, "%lu", (unsigned long) mum->header_from);
}

static int
mbox_get_body_transport (mu_stream_t is, mu_transport_t *pin,
			 mu_transport_t *pout)
{
  mu_body_t body = mu_stream_get_owner (is);
  mu_message_t msg = mu_body_get_owner (body);
  mbox_message_t mum = mu_message_get_owner (msg);
  return mbox_get_transport2 (mum, pin, pout);
}

static int
mbox_get_transport2 (mbox_message_t mum, mu_transport_t *pin,
		     mu_transport_t *pout)
{
  if (mum == NULL)
    return EINVAL;
  return mu_stream_get_transport2 (mum->mud->mailbox->stream, pin, pout);
}

static int
mbox_get_attr_flags (mu_attribute_t attr, int *pflags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  mbox_message_t mum = mu_message_get_owner (msg);

  if (mum == NULL)
    return EINVAL;
  if (pflags)
    *pflags = mum->attr_flags;
  return 0;
}

static int
mbox_set_attr_flags (mu_attribute_t attr, int flags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  mbox_message_t mum = mu_message_get_owner (msg);

  if (mum == NULL)
    return EINVAL;
  mum->attr_flags |= flags;
  return 0;
}

static int
mbox_unset_attr_flags (mu_attribute_t attr, int flags)
{
  mu_message_t msg = mu_attribute_get_owner (attr);
  mbox_message_t mum = mu_message_get_owner (msg);

  if (mum == NULL)
    return EINVAL;
  mum->attr_flags &= ~flags;
  return 0;
}

static int
mbox_body_readline (mu_stream_t is, char *buffer, size_t buflen,
		    mu_off_t off, size_t *pnread)
{
  mu_body_t body = mu_stream_get_owner (is);
  mu_message_t msg = mu_body_get_owner (body);
  mbox_message_t mum = mu_message_get_owner (msg);

  return mbox_readstream (mum, buffer, buflen, off, pnread, 1,
			  mum->body, mum->body_end);
}

static int
mbox_body_read (mu_stream_t is, char *buffer, size_t buflen,
		mu_off_t off, size_t *pnread)
{
  mu_body_t body = mu_stream_get_owner (is);
  mu_message_t msg = mu_body_get_owner (body);
  mbox_message_t mum = mu_message_get_owner (msg);
  return mbox_readstream (mum, buffer, buflen, off, pnread, 0,
			  mum->body, mum->body_end);
}

static int
mbox_readstream (mbox_message_t mum, char *buffer, size_t buflen,
		 mu_off_t off, size_t *pnread, int isreadline,
		 mu_off_t start, mu_off_t end)
{
  size_t nread = 0;
  int status = 0;

  if (buffer == NULL || buflen == 0)
    {
      if (pnread)
	*pnread = nread;
      return 0;
    }

  mu_monitor_rdlock (mum->mud->mailbox->monitor);
#ifdef WITH_PTHREAD
  /* read() is cancellation point since we're doing a potentially
     long operation.  Lets make sure we clean the state.  */
  pthread_cleanup_push (mbox_cleanup, (void *)mum->mud->mailbox);
#endif
  {
    mu_off_t ln = end - (start + off);
    if (ln > 0)
      {
	/* Position the file pointer and the buffer.  */
	nread = ((size_t)ln < buflen) ? (size_t)ln : buflen;
	if (isreadline)
	  status = mu_stream_readline (mum->mud->mailbox->stream,
				       buffer, buflen,
				       start + off, &nread);
	else
	  status = mu_stream_read (mum->mud->mailbox->stream, buffer, nread,
				   start + off, &nread);
      }
  }
  mu_monitor_unlock (mum->mud->mailbox->monitor);
#ifdef WITH_PTHREAD
  pthread_cleanup_pop (0);
#endif

  if (pnread)
    *pnread = nread;
  return status;
}

static int
mbox_header_fill (mu_header_t header, char *buffer, size_t len,
		  mu_off_t off, size_t *pnread)
{
  mu_message_t msg = mu_header_get_owner (header);
  mbox_message_t mum = mu_message_get_owner (msg);
  return mbox_readstream (mum, buffer, len, off, pnread, 0,
			  mum->header_from_end, mum->body);
}

static int
mbox_body_size (mu_body_t body, size_t *psize)
{
  mu_message_t msg = mu_body_get_owner (body);
  mbox_message_t mum = mu_message_get_owner (msg);
  if (mum == NULL)
    return EINVAL;
  if (psize)
    *psize = mum->body_end - mum->body;
  return 0;
}

static int
mbox_stream_size (mu_stream_t stream, mu_off_t *psize)
{
  mu_body_t body = mu_stream_get_owner (stream);
  return mbox_body_size (body, (size_t*) psize);
}

static int
mbox_body_lines (mu_body_t body, size_t *plines)
{
  mu_message_t msg = mu_body_get_owner (body);
  mbox_message_t mum = mu_message_get_owner (msg);
  if (mum == NULL)
    return EINVAL;
  if (plines)
    *plines = mum->body_lines;
  return 0;
}

static int
mbox_envelope_date (mu_envelope_t envelope, char *buf, size_t len,
		    size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  mbox_message_t mum = mu_message_get_owner (msg);
  size_t n = 0;
  int status;
  char buffer[512];
  char *s;

  if (mum == NULL)
    return EINVAL;

  status = mu_stream_readline (mum->mud->mailbox->stream,
			       buffer, sizeof(buffer),
			       mum->header_from, &n);
  if (status != 0)
    {
      if (pnwrite)
	*pnwrite = 0;
      return status;
    }

  /* Format:  "From [sender] [date]" */
  /* strlen ("From ") == 5 */
  if (n > 5 && (s = strchr (buffer + 5, ' ')) != NULL)
    {
      if (buf && len > 0)
	{
	  len--; /* Leave space for the null.  */
	  strncpy (buf, s + 1, len)[len] = '\0';
	  len = strlen (buf);
	}
      else
	len = strlen (s + 1);
    }
  else
    len = 0;

  if (pnwrite)
    *pnwrite = len;
  return 0;
}

static int
mbox_envelope_sender (mu_envelope_t envelope, char *buf, size_t len,
		      size_t *pnwrite)
{
  mu_message_t msg = mu_envelope_get_owner (envelope);
  mbox_message_t mum = mu_message_get_owner (msg);
  size_t n = 0;
  int status;
  char buffer[512];
  char *s;

  if (mum == NULL)
    return EINVAL;

  status = mu_stream_readline (mum->mud->mailbox->stream, buffer,
			       sizeof(buffer),
			       mum->header_from, &n);
  if (status != 0)
    {
      if (pnwrite)
	*pnwrite = 0;
      return status;
    }

  /* Format:  "From [sender] [date]" */
  /* strlen ("From ") == 5 */
  if (n > 5 && (s = strchr (buffer + 5, ' ')) != NULL)
    {
      /* Put a NULL to isolate the sender string, make a C string.  */
      *s = '\0';
      if (buf && len > 0)
	{
	  len--; /* leave space for the null */
	  strncpy (buf, buffer + 5, len)[len] = '\0';
	  len = strlen (buf);
	}
      else
	len = strlen (buffer + 5);
    }
  else
    len = 0;

  if (pnwrite)
    *pnwrite = len;
  return 0;
}

static int
new_message (mu_mailbox_t mailbox, mbox_message_t mum, mu_message_t *pmsg)
{
  int status;
  mu_message_t msg;

  /* Get an empty message struct.  */
  status = mu_message_create (&msg, mum);
  if (status != 0)
    return status;

  /* Set the header.  */
  {
    mu_header_t header = NULL;
    status = mu_header_create (&header, NULL, 0, msg);
    if (status != 0)
      {
	mu_message_destroy (&msg, mum);
	return status;
      }
    mu_header_set_fill (header, mbox_header_fill, msg);
    mu_message_set_header (msg, header, mum);
  }

  /* Set the attribute.  */
  {
    mu_attribute_t attribute;
    status = mu_attribute_create (&attribute, msg);
    if (status != 0)
      {
	mu_message_destroy (&msg, mum);
	return status;
      }
    mu_attribute_set_get_flags (attribute, mbox_get_attr_flags, msg);
    mu_attribute_set_set_flags (attribute, mbox_set_attr_flags, msg);
    mu_attribute_set_unset_flags (attribute, mbox_unset_attr_flags, msg);
    mu_message_set_attribute (msg, attribute, mum);
  }

  /* Prepare the body.  */
  {
    mu_body_t body = NULL;
    mu_stream_t stream = NULL;
    if ((status = mu_body_create (&body, msg)) != 0
	|| (status = mu_stream_create (&stream,
				       mailbox->flags | MU_STREAM_SEEKABLE,
				       body)) != 0)
      {
	mu_body_destroy (&body, msg);
	mu_stream_destroy (&stream, body);
	mu_message_destroy (&msg, mum);
	return status;
      }
    mu_stream_set_read (stream, mbox_body_read, body);
    mu_stream_set_readline (stream, mbox_body_readline, body);
    mu_stream_set_get_transport2 (stream, mbox_get_body_transport, body);
    mu_stream_set_size (stream, mbox_stream_size, body);
    mu_body_set_stream (body, stream, msg);
    mu_body_set_size (body, mbox_body_size, msg);
    mu_body_set_lines (body, mbox_body_lines, msg);
    mu_message_set_body (msg, body, mum);
  }

  /* Set the envelope.  */
  {
    mu_envelope_t envelope= NULL;
    status = mu_envelope_create (&envelope, msg);
    if (status != 0)
      {
	mu_message_destroy (&msg, mum);
	return status;
      }
    mu_envelope_set_sender (envelope, mbox_envelope_sender, msg);
    mu_envelope_set_date (envelope, mbox_envelope_date, msg);
    mu_message_set_envelope (msg, envelope, mum);
  }

  /* Set the UID.  */
  mu_message_set_uid (msg, mbox_message_uid, mum);
  mu_message_set_qid (msg, mbox_message_qid, mum);
  
  /* Attach the message to the mailbox mbox data.  */
  mum->message = msg;
  mu_message_set_mailbox (msg, mailbox, mum);

  *pmsg = msg;
  
  return 0;
}  

static int
mbox_get_message (mu_mailbox_t mailbox, size_t msgno, mu_message_t *pmsg)
{
  int status;
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum;

  /* Sanity checks.  */
  if (pmsg == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (mud == NULL)
    return EINVAL;

  /* If we did not start a scanning yet do it now.  */
  if (mud->messages_count == 0)
    {
      status = mbox_scan0 (mailbox, 1, NULL, 0);
      if (status != 0)
	return status;
    }

  /* Second sanity: check the message number.  */
  if (!(mud->messages_count > 0
	&& msgno > 0
	&& msgno <= mud->messages_count))
    return EINVAL;

  mum = mud->umessages[msgno - 1];

  /* Check if we already have it.  */
  if (mum->message)
    {
      if (pmsg)
	*pmsg = mum->message;
      return 0;
    }

  MU_DEBUG2 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_get_message (%s, %lu)\n",
	     mud->name, (unsigned long) msgno);

  return new_message (mailbox, mum, pmsg);
}

static int
qid2off (mu_message_qid_t qid, mu_off_t *pret)
{
  mu_off_t ret = 0;
  for (;*qid; qid++)
    {
      if (!('0' <= *qid && *qid <= '9'))
	return 1;
      ret = ret * 10 + *qid - '0';
    }
  *pret = ret;
  return 0;
}
      
static int
mbox_quick_get_message (mu_mailbox_t mailbox, mu_message_qid_t qid,
			mu_message_t *pmsg)
{
  int status;
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum;
  mu_off_t offset;
  
  if (mailbox == NULL || qid2off (qid, &offset)
      || !(mailbox->flags & MU_STREAM_QACCESS))
    return EINVAL;

  if (mud->messages_count == 0)
    {
      status = mbox_scan1 (mailbox, offset, 0);
      if (status != 0)
	return status;
      if (mud->messages_count == 0)
	return MU_ERR_NOENT;
    }

  /* Quick access mode retrieves only one message */
  mum = mud->umessages[0]; 

  /* Check if we already have it and verify if it is the right one. */
  if (mum->message)
    {
      char *vqid;
      status = mu_message_get_qid (mum->message, &vqid);
      if (status)
	return status;
      status = strcmp (qid, vqid);
      free (vqid);
      if (status)
	return MU_ERR_EXISTS;
      if (pmsg)
	*pmsg = mum->message;
      return 0;
    }

  return new_message (mailbox, mum, pmsg);
}
     
static int
mbox_append_message (mu_mailbox_t mailbox, mu_message_t msg)
{
  int status = 0;
  mbox_data_t mud = mailbox->data;
  mu_off_t qid;
  
  if (msg == NULL || mud == NULL)
    return EINVAL;

  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1, "mbox_append_message (%s)\n",
		  mud->name);

  switch (mud->state)
    {
    case MBOX_NO_STATE:
      if ((status = mu_locker_lock (mailbox->locker)) != 0)
	{
	  MU_DEBUG1 (mailbox->debug, MU_DEBUG_TRACE1,
			  "mbox_append_message: %s\n",
			  mu_strerror(status));
	  return status;
	}

    default:
      {
	mu_off_t size;
	/* Move to the end of the file, not necesary if _APPEND mode.  */
	if ((status = mu_stream_size (mailbox->stream, &size)) != 0
	    || (qid = size,
	        status = mbox_append_message0 (mailbox, msg,
					       &size, 0, 0)) != 0)
	  {
	    if (status != EAGAIN)
	      mu_locker_unlock (mailbox->locker);
	    return status;
	  }
      }
    }
  mu_locker_unlock (mailbox->locker);

  if (mailbox->observable)
    {
      char *buf = NULL;
      mu_asprintf (&buf, "%lu", (unsigned long) qid);
      mu_observable_notify (mailbox->observable, MU_EVT_MESSAGE_APPEND, buf);
      free (buf);
    }
  
  return 0;
}

int
restore_sender (mu_message_t msg, mbox_data_t mud)
{
  mu_header_t hdr;
  char *from = NULL;
  int rc = 0;
  
  if (mu_message_get_header (msg, &hdr) == 0)
    mu_header_aget_value (hdr, MU_HEADER_FROM, &from);

  if (from)
    {
      int status;
      mu_address_t addr;
      
      status = mu_address_create (&addr, from);
      free (from);
      from = NULL;
      if (status == 0)
	mu_address_aget_email (addr, 1, &from);
      mu_address_destroy (&addr);
    }

  if (!from)
    {
      from = strdup (PACKAGE);
      if (!from) 
	return ENOMEM;
    }

  mud->sender = strdup (from);
  if (!mud->sender)
    rc = ENOMEM;
  free (from);
  return rc;
}

int
restore_date (mu_message_t msg, mbox_data_t mud)
{
  mu_header_t hdr;
  char *date = NULL;
  time_t t;
  int rc = 0;
  
  if (mu_message_get_header (msg, &hdr) == 0)
    mu_header_aget_value (hdr, MU_HEADER_DATE, &date);

  if (date && mu_parse_date (date, &t, NULL))
    {
      char datebuf[MU_ENVELOPE_DATE_LENGTH+1];
      
      /* FIXME: 1. Preserve TZ info */
      mu_strftime (datebuf, sizeof datebuf, MU_ENVELOPE_DATE_FORMAT,
		   localtime (&t));
      free (date);
      date = strdup (datebuf);
    }
  else
    {
      time (&t);
      free (date);
      date = strdup (ctime (&t));
    }

  mud->date = date;
  if (!mud->date)
    rc = ENOMEM;
  return rc;
}

static int
write_array (mu_stream_t stream, mu_off_t *poff, int count, const char **array)
{
  int status;
  for (; count; count--, array++)
    {
      size_t len = strlen (*array);
      size_t i = 0, n;
      
      do
	{
	  status = mu_stream_write (stream, *array + i, len - i, *poff, &n);
	  if (status)
	    return status;
	  *poff += n;
	  i += n;
	}
      while (i < len);
    }
  return 0;
}


/* FIXME: Do we need to escape body line that begins with "From "? This
   will require reading the body line by line instead of by chunks,
   considerably hurting perfomance when expunging.  But should not this
   be the responsibility of the client ?  */
static int
mbox_append_message0 (mu_mailbox_t mailbox, mu_message_t msg, mu_off_t *psize,
		      int is_expunging, int first)
{
  mbox_data_t mud = mailbox->data;
  int status = 0;
  size_t n = 0;
  char nl = '\n';
  size_t orig_size = *psize;
  char *s;
  
  switch (mud->state)
    {
    case MBOX_NO_STATE:
      mud->off = 0;
      mud->state = MBOX_STATE_APPEND_SENDER;

    case MBOX_STATE_APPEND_SENDER:
      /* Generate the sender for the "From " separator.  */
      {
	char *s;
	mu_envelope_t envelope = NULL;
	mu_message_get_envelope (msg, &envelope);
	status = mu_envelope_aget_sender (envelope, &mud->sender);
	switch (status) {
	case 0:
	  break;
	  
	case EAGAIN:
	  return status;

	case MU_ERR_NOENT: /* Envelope headers not found: try to guess */
	  free (mud->sender);
          mud->sender = NULL;
	  status = restore_sender (msg, mud);
	  if (status == 0)
            break;
	  
	default:
	  free (mud->sender);
	  free (mud->date);
	  mud->date = mud->sender = NULL;
	  mud->state = MBOX_NO_STATE;
	  return status;
	}

	/* Nuke trailing newline.  */
	s = strchr (mud->sender, nl);
	if (s)
	  *s = '\0';
	mud->state = MBOX_STATE_APPEND_DATE;
      }

    case MBOX_STATE_APPEND_DATE:
      /* Generate a date for the "From "  separator.  */
      {
	mu_envelope_t envelope = NULL;
	const char *envarr[5];
	
	mu_message_get_envelope (msg, &envelope);
	status = mu_envelope_aget_date (envelope, &mud->date);
	switch (status) {
	case 0:
	  break;
	  
	case EAGAIN:
	  return status;

	case MU_ERR_NOENT: /* Envelope headers not found: try to guess */
	  free (mud->date);
          mud->date = NULL;
	  status = restore_date (msg, mud);
	  if (status == 0)
	    break;
	  
	default:
	  free (mud->sender);
	  free (mud->date);
	  mud->date = mud->sender = NULL;
	  mud->state = MBOX_NO_STATE;
	  return status;
	}

	/* Nuke trailing newline.  */
	s = strchr (mud->date, nl);
	if (s)
	  *s = '\0';

	/* Write the separator to the mailbox.  */
	envarr[0] = "From ";
	envarr[1] = mud->sender;
	envarr[2] = " ";
	envarr[3] = mud->date;
	envarr[4] = "\n";
	
	status = write_array (mailbox->stream, psize, 5, envarr);
	if (status)
	  break;

	free (mud->sender);
	free (mud->date);
	mud->sender = mud->date = NULL;
	/* If we are not expunging get the message in one block via the stream
	   message instead of the header/body.  This is good for POP where
	   there is no separation between header and body(RETR).  */
	if (! is_expunging)
	  {
	    mud->state = MBOX_STATE_APPEND_MESSAGE;
	    break;
	  }
	mud->state = MBOX_STATE_APPEND_HEADER;
      }

    case MBOX_STATE_APPEND_HEADER:
      /* Append the Header.  */
      {
	char buffer[1024];
	size_t nread = 0;
	mu_stream_t is = NULL;
	mu_header_t header = NULL;
	mu_message_get_header (msg, &header);
	mu_header_get_stream (header, &is);
	do
	  {
	    status = mu_stream_readline (is, buffer, sizeof (buffer), mud->off,
					 &nread);
	    if (status != 0)
	      {
		if (status != EAGAIN)
		  {
		    mud->state = MBOX_NO_STATE;
		    mud->off = 0;
		  }
		mu_stream_truncate (mailbox->stream, orig_size);
		return status;
	      }
	    mud->off += nread;
	    if (*buffer == '\n')
	      break;

	    /* We do not copy the Status since it is rewritten by the
	       attribute code below. Ditto for X-UID and X-IMAPBase.
	       FIXME:
	       - We have a problem here the header may not fit the buffer.
	       - Should  we skip the IMAP "X-Status"? */
	    if ((mu_c_strncasecmp (buffer, "Status", 6) == 0)
		|| (mu_c_strncasecmp (buffer, "X-IMAPbase", 10) == 0)
		/* FIXME: isn't the length of "X-UID" 5, not 4? And
		 this will match X-UID and X-UIDL, is this intended? */
		|| (mu_c_strncasecmp (buffer, "X-UID", 4) == 0
		    && (buffer[5] == ':' || buffer[5] == ' '
			|| buffer[5] == '\t')))
	      continue;

	    status = mu_stream_write (mailbox->stream, buffer, nread,
				      *psize, &n);
	    if (status)
	      break;
	    *psize += n;
	  }
	while (nread > 0);
	mud->off = 0;

	/* Rewrite the X-IMAPbase marker. */
	if (first && is_expunging)
	  {
	    n = sprintf (buffer, "X-IMAPbase: %lu %u\n",
			 (unsigned long) mud->uidvalidity,
			 (unsigned) mud->uidnext);
	    status = mu_stream_write (mailbox->stream, buffer, n, *psize, &n);
	    if (status)
	      break;
	    *psize += n;
	  }
	mud->state = MBOX_STATE_APPEND_ATTRIBUTE;
      }

      case MBOX_STATE_APPEND_ATTRIBUTE:
      /* Put the new attributes.  */
      {
#define STATUS_PREFIX_LEN (sizeof(MU_HEADER_STATUS) - 1 + 2)
	char abuf[STATUS_PREFIX_LEN + MU_STATUS_BUF_SIZE + 1];
	size_t na = 0;
	mu_attribute_t attr = NULL;

	strcpy(abuf, MU_HEADER_STATUS);
	strcat(abuf, ": ");
	mu_message_get_attribute (msg, &attr);
	mu_attribute_to_string (attr, abuf + STATUS_PREFIX_LEN, 
	                        sizeof(abuf) - STATUS_PREFIX_LEN - 1, &na);
	strcat (abuf, "\n");
	na = strlen (abuf);
	mu_stream_write (mailbox->stream, abuf, na, *psize, &n);
	if (status != 0)
	  break;
	*psize += n;

	mud->state = MBOX_STATE_APPEND_UID;
      }

    case MBOX_STATE_APPEND_UID:
      /* The new X-UID. */
      {
	char suid[64];
	size_t uid = 0;
	if (is_expunging)
	  {
	    status = mu_message_get_uid (msg, &uid);
	    if (status == EAGAIN)
	      return status;
	  }
	else
	  uid = mud->uidnext++;

	if (status == 0 || uid != 0)
	  {
	    n = sprintf (suid, "X-UID: %u\n", (unsigned) uid);
	    /* Put the UID.  */
	    status = mu_stream_write (mailbox->stream, suid, n, *psize, &n);
	    if (status)
	      break;
	    *psize += n;
	  }

	/* New line separator of the Header.  */
	status = mu_stream_write (mailbox->stream, &nl , 1, *psize, &n);
	if (status)
	  break;
	*psize += n;
	mud->state = MBOX_STATE_APPEND_BODY;
      }

    case MBOX_STATE_APPEND_BODY:
      /* Append the Body.  */
      {
	char buffer[1024];
	size_t nread = 0;
	mu_stream_t is = NULL;
	mu_body_t body = NULL;
	mu_message_get_body (msg, &body);
	mu_body_get_stream (body, &is);
	do
	  {
	    status = mu_stream_read (is, buffer, sizeof (buffer), mud->off,
				     &nread);
	    if (status != 0)
	      {
		if (status != EAGAIN)
		  {
		    mud->state = MBOX_NO_STATE;
		    mud->off = 0;
		  }
		return status;
	      }
	    mud->off += nread;
	    status = mu_stream_write (mailbox->stream, buffer, nread,
				      *psize, &n);
	    if (status)
	      break;
	    *psize += n;
	  }
	while (nread > 0);
	mud->off = 0;
	n = 0;
	status = mu_stream_write (mailbox->stream, &nl, 1, *psize, &n);
	if (status)
	  break;
	*psize += n;
      }

    default:
      break;
    }

  /* If not expunging we are taking the stream message.  */
  if (!is_expunging)
    {
      switch (mud->state)
        {
	case MBOX_STATE_APPEND_MESSAGE:
	  {
	    /* Append the Message.  */
	    char buffer[1024];
	    size_t nread = 0;
	    mu_stream_t is = NULL;
	    mu_message_get_stream (msg, &is);
	    do
	      {
		status = mu_stream_read (is, buffer, sizeof (buffer), mud->off,
				      &nread);
		if (status != 0)
		  {
		    if (status != EAGAIN)
		      {
			mud->state = MBOX_NO_STATE;
			mud->off = 0;
		      }
		    mu_stream_truncate (mailbox->stream, orig_size);
		    return status;
		  }
		status = mu_stream_write (mailbox->stream, buffer, nread,
					  *psize, &n);
		if (status)
		  break;
		mud->off += nread;
		*psize += n;
	      }
	    while (nread > 0);
	    if (status)
	      break;
       	    status = mu_stream_write (mailbox->stream, &nl, 1, *psize, &n);
	    if (status == 0)
	      *psize += n;
	  }

	default:
	  break;
	}
    } /* is_expunging */
  mud->state = MBOX_NO_STATE;
  if (status)
    mu_stream_truncate (mailbox->stream, orig_size);
  else
    mu_stream_flush (mailbox->stream);
  return status;
}

static int
mbox_get_size (mu_mailbox_t mailbox, mu_off_t *psize)
{
  mu_off_t size;
  int status;

  /* Maybe was not open yet ??  */
  status  = mu_stream_size (mailbox->stream, &size);
  if (status != 0)
    return status;
  if (psize)
    *psize = size;
  return 0;
}

static int
mbox_messages_count (mu_mailbox_t mailbox, size_t *pcount)
{
  mbox_data_t mud = mailbox->data;

  if (mud == NULL)
    return EINVAL;

  if (! mbox_is_updated (mailbox))
    return mbox_scan0 (mailbox,  mud->messages_count, pcount, 0);

  if (pcount)
    *pcount = mud->messages_count;

  return 0;
}

/* A "recent" message is the one not marked with MU_ATTRIBUTE_SEEN
   ('O' in the Status header), i.e. a message that is first seen
   by the current session (see attributes.h) */
static int
mbox_messages_recent (mu_mailbox_t mailbox, size_t *pcount)
{
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum;
  size_t j, recent;

  /* If we did not start a scanning yet do it now.  */
  if (mud->messages_count == 0)
    {
      int status = mbox_scan0 (mailbox, 1, NULL, 0);
      if (status != 0)
	return status;
    }
  for (recent = j = 0; j < mud->messages_count; j++)
    {
      mum = mud->umessages[j];
      if (mum && MU_ATTRIBUTE_IS_UNSEEN(mum->attr_flags))
	recent++;
    }
  *pcount = recent;
  return 0;
}

/* An "unseen" message is the one that has not been read yet */
static int
mbox_message_unseen (mu_mailbox_t mailbox, size_t *pmsgno)
{
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum;
  size_t j, unseen;

  /* If we did not start a scanning yet do it now.  */
  if (mud->messages_count == 0)
    {
      int status = mbox_scan0 (mailbox, 1, NULL, 0);
      if (status != 0)
	return status;
    }
  for (unseen = j = 0; j < mud->messages_count; j++)
    {
      mum = mud->umessages[j];
      if (mum && MU_ATTRIBUTE_IS_UNREAD(mum->attr_flags))
	{
	  unseen = j + 1;
	  break;
	}
    }
  *pmsgno = unseen;
  return 0;
}

static int
mbox_uidvalidity (mu_mailbox_t mailbox, unsigned long *puidvalidity)
{
  mbox_data_t mud = mailbox->data;
  int status = mbox_messages_count (mailbox, NULL);
  if (status != 0)
    return status;
  /* If we did not start a scanning yet do it now.  */
  if (mud->messages_count == 0)
    {
      status = mbox_scan0 (mailbox, 1, NULL, 0);
      if (status != 0)
	return status;
    }
  if (puidvalidity)
    *puidvalidity = mud->uidvalidity;
  return 0;
}

static int
mbox_uidnext (mu_mailbox_t mailbox, size_t *puidnext)
{
  mbox_data_t mud = mailbox->data;
  int status = mbox_messages_count (mailbox, NULL);
  if (status != 0)
    return status;
  /* If we did not start a scanning yet do it now.  */
  if (mud->messages_count == 0)
    {
      status = mbox_scan0 (mailbox, 1, NULL, 0);
      if (status != 0)
	return status;
    }
  if (puidnext)
    *puidnext = mud->uidnext;
  return 0;
}

#ifdef WITH_PTHREAD
void
mbox_cleanup (void *arg)
{
  mu_mailbox_t mailbox = arg;
  mu_monitor_unlock (mailbox->monitor);
  mu_locker_unlock (mailbox->locker);
}
#endif
