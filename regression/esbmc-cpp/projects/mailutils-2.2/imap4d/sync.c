/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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
#include <mailutils/observer.h>

/*

 */
struct _uid_table
{
  size_t uid;
  size_t msgno;
  int notify;
  mu_attribute_t attr;
};

static struct _uid_table *uid_table;
static size_t uid_table_count;
static int uid_table_loaded;

static void
add_flag (char **pbuf, const char *f)
{
  char *abuf = *pbuf;
  abuf = realloc (abuf, strlen (abuf) + strlen (f) + 2);
  if (abuf == NULL)
    imap4d_bye (ERR_NO_MEM);
  if (*abuf)
    strcat (abuf, " ");
  strcat (abuf, "\\Seen");
  *pbuf = abuf;
}

static void
notify_flag (size_t msgno, mu_attribute_t oattr)
{
  mu_message_t msg = NULL;
  mu_attribute_t nattr = NULL;
  int status ;
  mu_mailbox_get_message (mbox, msgno, &msg);
  mu_message_get_attribute (msg, &nattr);
  status = mu_attribute_is_equal (oattr, nattr);

  if (status == 0)
    {
      char *abuf = malloc (1);;
      if (!abuf)
	imap4d_bye (ERR_NO_MEM);
      *abuf = '\0';
      if (mu_attribute_is_seen (nattr) && mu_attribute_is_read (nattr))
	if (!mu_attribute_is_seen (oattr) && !mu_attribute_is_read (oattr))
	  {
	    mu_attribute_set_seen (oattr);
	    mu_attribute_set_read (oattr);
	    add_flag (&abuf, "\\Seen");
	  }
      if (mu_attribute_is_answered (nattr))
	if (!mu_attribute_is_answered (oattr))
	  {
	    mu_attribute_set_answered (oattr);
	    add_flag (&abuf, "\\Answered");
	  }
      if (mu_attribute_is_flagged (nattr))
	if (!mu_attribute_is_flagged (oattr))
	  {
	    mu_attribute_set_flagged (oattr);
	    add_flag (&abuf, "\\Flagged");
	  }
      if (mu_attribute_is_deleted (nattr))
	if (!mu_attribute_is_deleted (oattr))
	  {
	    mu_attribute_set_deleted (oattr);
	    add_flag (&abuf, "\\Deleted");
	  }
      if (mu_attribute_is_draft (nattr))
	if (!mu_attribute_is_draft (oattr))
	  {
	    mu_attribute_set_draft (oattr);
	    add_flag (&abuf, "\\Draft");
	  }
      if (mu_attribute_is_recent (nattr))
	if (!mu_attribute_is_recent (oattr))
	  {
	    mu_attribute_set_recent (oattr);
	    add_flag (&abuf, "\\Recent");
	  }
      if (*abuf)
	util_out (RESP_NONE, "%lu FETCH FLAGS (%s)",
		  (unsigned long) msgno, abuf);
      free (abuf);
    }
}

/* The EXPUNGE response reports that the specified message sequence
   number has been permanently removed from the mailbox.  The message
   sequence number for each successive message in the mailbox is
   immediately decremented by 1, and this decrement is reflected in
   message sequence numbers in subsequent responses (including other
   untagged EXPUNGE responses). */
static void
notify_deleted (void)
{
  if (uid_table)
    {
      size_t i;
      size_t decr = 0;
      for (i = 0; i < uid_table_count; i++)
	{
	  if (!(uid_table[i].notify))
	    {
	      util_out (RESP_NONE, "%lu EXPUNGED",
			(unsigned long) uid_table[i].msgno-decr);
	      uid_table[i].notify = 1;
	      decr++;
	    }
	}
    }
}


static int
notify_uid (size_t uid)
{
  if (uid_table)
    {
      size_t i;
      for (i = 0; i < uid_table_count; i++)
	{
	  if (uid_table[i].uid == uid)
	    {
	      notify_flag (uid_table[i].msgno, uid_table[i].attr);
	      uid_table[i].notify = 1;
	      return 1;
	    }
	}
    }
  return 0;
}

static void
free_uids (void)
{
  if (uid_table)
    {
      size_t i;
      for (i = 0; i < uid_table_count; i++)
	mu_attribute_destroy (&(uid_table[i].attr), NULL);
      free (uid_table);
      uid_table = NULL;
    }
  uid_table_count = 0;
  uid_table_loaded = 0;
}

static void
reset_notify (void)
{
  size_t i;

  for (i = 0; i < uid_table_count; i++)
    uid_table[i].notify = 0;
}

static void
reset_uids (void)
{
  size_t total = 0;
  size_t i;

  free_uids ();

  mu_mailbox_messages_count (mbox, &total);
  for (i = 1; i <= total; i++)
    {
      mu_message_t msg = NULL;
      mu_attribute_t attr = NULL;
      size_t uid = 0;
      uid_table = realloc (uid_table, sizeof (*uid_table) *
			   (uid_table_count + 1));
      if (!uid_table)
	imap4d_bye (ERR_NO_MEM);
      mu_mailbox_get_message (mbox, i, &msg);
      mu_message_get_attribute (msg, &attr);
      mu_message_get_uid (msg, &uid);
      uid_table[uid_table_count].uid = uid;
      uid_table[uid_table_count].msgno = i;
      uid_table[uid_table_count].notify = 0;
      mu_attribute_create (&(uid_table[uid_table_count].attr), NULL);
      mu_attribute_copy (uid_table[uid_table_count].attr, attr);
      uid_table_count++;
    }
  uid_table_loaded = 1;
}

static void
notify (void)
{
  size_t total = 0;
  int reset = 0;
  size_t recent = 0;
  
  mu_mailbox_messages_count (mbox, &total);

  if (!uid_table)
    {
      reset = 1;
      reset_uids ();
    }
  
  if (uid_table)
    {
      size_t i;

      for (i = 1; i <= total; i++)
	{
	  mu_message_t msg = NULL;
	  size_t uid = 0;
	  mu_mailbox_get_message (mbox, i, &msg);
	  mu_message_get_uid (msg, &uid);
	  notify_uid (uid);
	}
      notify_deleted ();
      mu_mailbox_messages_recent (mbox, &recent);
    }

  util_out (RESP_NONE, "%lu EXISTS", (unsigned long) total);
  util_out (RESP_NONE, "%lu RECENT", (unsigned long) recent);

  if (!reset)
    reset_uids ();
  else
    reset_notify ();
}

size_t
uid_to_msgno (size_t uid)
{
  size_t i;
  for (i = 0; i < uid_table_count; i++)
    if (uid_table[i].uid == uid)
      return uid_table[i].msgno;
  return 0;
}

int
imap4d_sync_flags (size_t msgno)
{
  size_t i;
  for (i = 0; i < uid_table_count; i++)
    if (uid_table[i].msgno == msgno)
      {
	mu_message_t msg = NULL;
	mu_attribute_t attr = NULL;
	mu_mailbox_get_message (mbox, msgno, &msg);
	mu_message_get_attribute (msg, &attr);
	mu_attribute_copy (uid_table[i].attr, attr);
	break;
      }
  return 0;
}

static int mailbox_corrupt;

static int
action (mu_observer_t observer, size_t type, void *data, void *action_data)
{
  switch (type)
    {
    case MU_EVT_MAILBOX_CORRUPT:
      mailbox_corrupt = 1;
      break;

    case MU_EVT_MAILBOX_DESTROY:
      mailbox_corrupt = 0;
      break;
    }
  return 0;
}

void
imap4d_set_observer (mu_mailbox_t mbox)
{
  mu_observer_t observer;
  mu_observable_t observable;
      
  mu_observer_create (&observer, mbox);
  mu_observer_set_action (observer, action, mbox);
  mu_mailbox_get_observable (mbox, &observable);
  mu_observable_attach (observable, MU_EVT_MAILBOX_CORRUPT|MU_EVT_MAILBOX_DESTROY,
		     observer);
  mailbox_corrupt = 0;
}

int
imap4d_sync (void)
{
  /* If mbox --> NULL, it means to free all the resources.
     It may be because of close or before select/examine a new mailbox.
     If it was a close we do not send any notification.  */
  if (mbox == NULL)
    free_uids ();
  else if (!uid_table_loaded || !mu_mailbox_is_updated (mbox))
    {
      if (mailbox_corrupt)
	{
	  /* Some messages have been deleted from the mailbox by some other
	     party */
	  int status = mu_mailbox_close (mbox);
	  if (status)
	    imap4d_bye (ERR_MAILBOX_CORRUPTED);
	  status = mu_mailbox_open (mbox, MU_STREAM_RDWR);
	  if (status)
	    imap4d_bye (ERR_MAILBOX_CORRUPTED);
	  imap4d_set_observer (mbox);
	  free_uids ();
	  mailbox_corrupt = 0;
	  util_out (RESP_NONE,
		    "OK [ALERT] Mailbox modified by another program");
	}
      notify ();
    }
  else
    {
      size_t count = 0;
      mu_mailbox_messages_count (mbox, &count);
      if (count != uid_table_count)
	notify ();
    }
  return 0;
}
