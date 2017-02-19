/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
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

#define MAX_OPEN_STREAMS 16

/* Notifications ADD_MESG. */
#define DISPATCH_ADD_MSG(mbox,mhd,n)					\
  do									\
    {									\
      int bailing = 0;							\
      mu_monitor_unlock (mbox->monitor);				\
      if (mbox->observable)						\
	{								\
	  size_t tmp = n;						\
	  bailing = mu_observable_notify (mbox->observable,		\
					  MU_EVT_MESSAGE_ADD,		\
					  &tmp);			\
	}								\
      if (bailing != 0)							\
	{								\
	  if (pcount)							\
	    *pcount = (mhd)->msg_count;					\
	  mu_locker_unlock (mbox->locker);				\
	  return EINTR;							\
	}								\
      mu_monitor_wrlock (mbox->monitor);				\
} while (0);

#define MU_AMD_SIZE_FILE_NAME ".mu-size"

struct _amd_data;
struct _amd_message
{
  mu_stream_t stream;       /* Associated file stream */
  mu_off_t body_start;      /* Offset of body start in the message file */
  mu_off_t body_end;        /* Offset of body end (size of file, effectively)*/

  int orig_flags;           /* Original attribute flags */
  int attr_flags;           /* Current attribute flags */
	

  time_t mtime;             /* Time of last modification */
  size_t header_lines;      /* Number of lines in the header part */
  size_t body_lines;        /* Number of lines in the body */

  mu_message_t message;     /* Corresponding mu_message_t */
  struct _amd_data *amd;    /* Back pointer.  */
};

struct _amd_data
{
  size_t msg_size;               /* Size of struct _amd_message */
  int (*create) (struct _amd_data *, int flags);	
  int (*msg_init_delivery) (struct _amd_data *, struct _amd_message *);
  int (*msg_finish_delivery) (struct _amd_data *, struct _amd_message *,
			      const mu_message_t);
  void (*msg_free) (struct _amd_message *);
  int (*cur_msg_file_name) (struct _amd_message *, char **);	
  int (*new_msg_file_name) (struct _amd_message *, int, int, char **);
  int (*scan0)     (mu_mailbox_t mailbox, size_t msgno, size_t *pcount,
		    int do_notify);
  int (*mailbox_size) (mu_mailbox_t mailbox, mu_off_t *psize);
  int (*qfetch)    (struct _amd_data *, mu_message_qid_t qid);
  int (*msg_cmp) (struct _amd_message *, struct _amd_message *);
  int (*message_uid) (mu_message_t msg, size_t *puid);
  size_t (*next_uid) (struct _amd_data *mhd);
  
  /* List of messages: */
  size_t msg_count; /* number of messages in the list */
  size_t msg_max;   /* maximum message buffer capacity */
  struct _amd_message **msg_array;

  unsigned long uidvalidity;
  int has_new_msg;  /* New messages have been appended */
  char *name;                    /* Directory name */

  /* Pool of open message streams */
  struct _amd_message *msg_pool[MAX_OPEN_STREAMS];
  int pool_first;    /* Index to the first used entry in msg_pool */
  int pool_last;     /* Index to the first free entry in msg_pool */

  time_t mtime;      /* Time of last modification */

  mu_mailbox_t mailbox; /* Back pointer. */
};


int amd_init_mailbox (mu_mailbox_t mailbox, size_t mhd_size,
		      struct _amd_data **pmhd);
int _amd_message_insert (struct _amd_data *mhd, struct _amd_message *msg);
int amd_message_stream_open (struct _amd_message *mhm);
void amd_message_stream_close (struct _amd_message *mhm);
void amd_cleanup (void *arg);
struct _amd_message *_amd_get_message (struct _amd_data *amd, size_t msgno);
int amd_msg_lookup (struct _amd_data *amd, struct _amd_message *msg,
		    size_t *pret);
