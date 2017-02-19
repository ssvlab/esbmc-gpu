/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2003, 2005, 2007, 2010 Free Software Foundation, Inc.

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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>

#ifdef WITH_PTHREAD
# ifdef HAVE_PTHREAD_H
#  ifndef _XOPEN_SOURCE
#   define _XOPEN_SOURCE  500
#  endif
#  include <pthread.h>
# endif
#endif

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailbox0.h>
#include <registrar0.h>

#include <mailutils/address.h>
#include <mailutils/attribute.h>
#include <mailutils/body.h>
#include <mailutils/debug.h>
#include <mailutils/envelope.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/header.h>
#include <mailutils/locker.h>
#include <mailutils/message.h>
#include <mailutils/mutil.h>
#include <mailutils/nls.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/stream.h>
#include <mailutils/url.h>

struct _mbox_message;
struct _mbox_data;

typedef struct _mbox_data* mbox_data_t;
typedef struct _mbox_message* mbox_message_t;

/* Keep the file positions of where the headers and bodies start and end.
   attr_flags is the "Status:" message.  */
struct _mbox_message
{
  /* Offset of the messages in the mailbox.  */
  off_t header_from;
  off_t header_from_end;
  off_t body;
  off_t body_end;

  size_t uid; /* IMAP uid.  */

  int attr_flags; /* The attr_flags contains the "Status:" attribute  */

  size_t header_lines;
  size_t body_lines;

  mu_message_t message; /* A message attach to it.  */
  mbox_data_t mud; /* Back pointer.  */
};

/* The umessages is an array of pointers that contains umessages_count of
   mbox_message_t*; umessages[umessages_count].  We do it this way because
   realloc() can move everything to a new memory region and invalidate all
   the pointers.  Thanks to <Dave Inglis> for pointing this out.  The
   messages_count is the count number of messages parsed so far.  */
struct _mbox_data
{
  mbox_message_t *umessages; /* Array.  */
  size_t umessages_count; /* How big is the umessages[].  */
  size_t messages_count; /* How many valid entry in umessages[].  */
  off_t size; /* Size of the mailbox.  */
  unsigned long uidvalidity;
  size_t uidnext;
  char *name;

  /* The variables below are use to hold the state when appending messages.  */
  enum mbox_state
  {
    MBOX_NO_STATE = 0,
    MBOX_STATE_APPEND_SENDER, MBOX_STATE_APPEND_DATE, MBOX_STATE_APPEND_HEADER,
    MBOX_STATE_APPEND_ATTRIBUTE, MBOX_STATE_APPEND_UID, MBOX_STATE_APPEND_BODY,
    MBOX_STATE_APPEND_MESSAGE
  } state ;
  char *sender;
  char *date;
  off_t off;
  mu_mailbox_t mailbox; /* Back pointer. */
};

int mbox_scan0 (mu_mailbox_t mailbox, size_t msgno, size_t *pcount,
		int do_notif);
int mbox_scan1 (mu_mailbox_t mailbox, mu_off_t offset, int do_notif);

#ifdef WITH_PTHREAD
void mbox_cleanup (void *arg);
#endif

