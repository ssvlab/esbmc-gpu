/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
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

/* First draft by Sergey Poznyakoff */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_MH

#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>

#ifdef WITH_PTHREAD
# ifdef HAVE_PTHREAD_H
#  ifndef _XOPEN_SOURCE
#   define _XOPEN_SOURCE  500
#  endif
#  include <pthread.h>
# endif
#endif

#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/attribute.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/envelope.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/locker.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>
#include <mailutils/observer.h>
#include <mailutils/io.h>
#include <mailutils/cctype.h>
#include <mailbox0.h>
#include <registrar0.h>
#include <amd.h>

struct _mh_message
{
  struct _amd_message amd_message;
  size_t seq_number;        /* message sequence number */
};

static int
mh_message_cmp (struct _amd_message *a, struct _amd_message *b)
{
  struct _mh_message *ma = (struct _mh_message *) a;
  struct _mh_message *mb = (struct _mh_message *) b;
  if (ma->seq_number < mb->seq_number)
    return -1;
  else if (ma->seq_number > mb->seq_number)
    return 1;
  return 0;
}

static size_t
_mh_next_seq (struct _amd_data *amd)
{
  struct _mh_message *msg = (struct _mh_message *)
                              _amd_get_message (amd, amd->msg_count);
  return (msg ? msg->seq_number : 0) + 1;
}

/* Return current filename for the message.
   NOTE: Allocates memory. */
static int
_mh_cur_message_name (struct _amd_message *amsg, char **pname)
{
  int status = 0;
  struct _mh_message *mhm = (struct _mh_message *) amsg;
  char *filename;
  char *pnum;
  size_t len;

  status = mu_asprintf (&pnum, "%lu", (unsigned long) mhm->seq_number);
  if (status)
    return status;
  len = strlen (amsg->amd->name) + 1 + strlen (pnum) + 1;
  filename = malloc (len);
  if (filename)
    {
      strcpy (filename, amsg->amd->name);
      strcat (filename, "/");
      strcat (filename, pnum);
      *pname = filename;
    }
  else
    status = ENOMEM;
  free (pnum);
  return status;
}

/* Return newfilename for the message.
   NOTE: Allocates memory. */
static int
_mh_new_message_name (struct _amd_message *amsg, int flags,
		      int expunge MU_ARG_UNUSED,
		      char **pname)
{
  int status = 0;
  struct _mh_message *mhm = (struct _mh_message *) amsg;
  char *filename;
  char *pnum;
  size_t len;

  status = mu_asprintf (&pnum, "%lu", (unsigned long) mhm->seq_number);
  if (status)
    return status;
  len = strlen (amsg->amd->name) + 1 +
               ((flags & MU_ATTRIBUTE_DELETED) ? 1 : 0) + strlen (pnum) + 1;
  filename = malloc (len);
  if (filename)
    {
      strcpy (filename, amsg->amd->name);
      strcat (filename, "/");
      if (flags & MU_ATTRIBUTE_DELETED)
	strcat (filename, ",");
      strcat (filename, pnum);
      *pname = filename;
    }
  else
    status = ENOMEM;
  free (pnum);
  return status;
}

/* Find the message with the given sequence number */
static struct _mh_message *
_mh_get_message_seq (struct _amd_data *amd, size_t seq)
{
  struct _mh_message msg;
  size_t index;
  
  msg.seq_number = seq;
  if (amd_msg_lookup (amd, (struct _amd_message*) &msg, &index))
    return NULL;

  return (struct _mh_message *) _amd_get_message (amd, index);
}

/* Scan the mailbox */
static int
mh_scan0 (mu_mailbox_t mailbox, size_t msgno MU_ARG_UNUSED, size_t *pcount, 
          int do_notify)
{
  struct _amd_data *amd = mailbox->data;
  struct _mh_message *msg;
  DIR *dir;
  struct dirent *entry;
  int status = 0;
  struct stat st;

  if (amd == NULL)
    return EINVAL;

  dir = opendir (amd->name);
  if (!dir)
    return errno;

  mu_monitor_wrlock (mailbox->monitor);

#ifdef WITH_PTHREAD
  pthread_cleanup_push (amd_cleanup, (void *)mailbox);
#endif

  mu_locker_lock (mailbox->locker);

  /* Do actual work. */

  while ((entry = readdir (dir)))
    {
      char *namep;
      int attr_flags;
      size_t num;

      attr_flags = 0;
      switch (entry->d_name[0])
	{
	case '.':
	  /* FIXME: .mh_sequences */
	  continue;
	case ',':
	  continue;
#if 0
	  attr_flags |= MU_ATTRIBUTE_DELETED;
	  namep = entry->d_name+1;
	  break;
#endif
	case '0':case '1':case '2':case '3':case '4':
	case '5':case '6':case '7':case '8':case '9':
	  namep = entry->d_name;
	  break;
	default:
	  /*FIXME: Invalid entry. Report? */
	  continue;
	}

      num = strtoul (namep, &namep, 10);
      if (namep[0])
	continue;

      msg = _mh_get_message_seq (amd, num);
      if (!msg)
	{
	  msg = calloc (1, sizeof(*msg));

	  msg->seq_number = num;
	  msg->amd_message.attr_flags = attr_flags;
	  msg->amd_message.orig_flags = msg->amd_message.attr_flags;

	  _amd_message_insert (amd, (struct _amd_message*) msg);
	}
      else
	{
	  msg->amd_message.attr_flags = attr_flags;
	  msg->amd_message.orig_flags = msg->amd_message.attr_flags;
	}
    }

  closedir (dir);

  if (do_notify)
    {
      size_t i;

      for (i = 0; i < amd->msg_count; i++)
	{
	  DISPATCH_ADD_MSG (mailbox, amd, i);
	}
    }
  
  if (stat (amd->name, &st) == 0)
    amd->mtime = st.st_mtime;

  if (pcount)
    *pcount = amd->msg_count;

  /* Reset the uidvalidity.  */
  if (amd->msg_count > 0)
    {
      if (amd->uidvalidity == 0)
	{
	  amd->uidvalidity = (unsigned long)time (NULL);
	  /* Tell that we have been modified for expunging.  */
	  if (amd->msg_count)
	    {
	      amd_message_stream_open (amd->msg_array[0]);
	      amd_message_stream_close (amd->msg_array[0]);
	      amd->msg_array[0]->attr_flags |= MU_ATTRIBUTE_MODIFIED;
	    }
	}
    }

  /* Clean up the things */

  amd_cleanup (mailbox);
#ifdef WITH_PTHREAD
  pthread_cleanup_pop (0);
#endif
  return status;
}

static int
mh_qfetch (struct _amd_data *amd, mu_message_qid_t qid)
{
  char *p;
  size_t num = 0;
  int attr_flags = 0;
  struct _mh_message *msg;

  p = qid + strlen (qid) - 1;
  if (!mu_isdigit (*p))
    return EINVAL;
  
  for (p--; p >= qid && mu_isdigit (*p); p--)
    ;

  if (p == qid)
    return EINVAL;

  num = strtoul (p + 1, NULL, 10);
  
  if (*p == ',')
    {
      attr_flags |= MU_ATTRIBUTE_DELETED;
      p--;
    }

  if (*p != '/')
    return EINVAL;
  
  msg = calloc (1, sizeof (*msg));
  msg->seq_number = num;
  msg->amd_message.attr_flags = attr_flags;
  msg->amd_message.orig_flags = msg->amd_message.attr_flags;
  _amd_message_insert (amd, (struct _amd_message*) msg);
  return 0;
}


/* Note: In this particular implementation the message sequence number
   serves also as its UID. This allows to avoid many problems related
   to keeping the uids in the headers of the messages. */

static int
mh_message_uid (mu_message_t msg, size_t *puid)
{
  struct _mh_message *mhm = mu_message_get_owner (msg);
  if (puid)
    *puid = mhm->seq_number;
  return 0;
}

static int
_mh_msg_init (struct _amd_data *amd, struct _amd_message *amm)
{
  struct _mh_message *mhm = (struct _mh_message *) amm;
  mhm->seq_number = _mh_next_seq (amd);
  return 0;
}



int
_mailbox_mh_init (mu_mailbox_t mailbox)
{
  int rc;
  struct _amd_data *amd;

  rc = amd_init_mailbox (mailbox, sizeof (struct _amd_data), &amd);
  if (rc)
    return rc;

  amd->msg_size = sizeof (struct _mh_message);
  amd->msg_free = NULL;
  amd->msg_init_delivery = _mh_msg_init;
  amd->msg_finish_delivery = NULL;
  amd->cur_msg_file_name = _mh_cur_message_name;
  amd->new_msg_file_name = _mh_new_message_name;
  amd->scan0 = mh_scan0;
  amd->qfetch = mh_qfetch;
  amd->msg_cmp = mh_message_cmp;
  amd->message_uid = mh_message_uid;
  amd->next_uid = _mh_next_seq;
  
  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailbox_get_property (mailbox, &property);
    mu_property_set_value (property, "TYPE", "MH", 1);
  }

  return 0;
}

#endif
