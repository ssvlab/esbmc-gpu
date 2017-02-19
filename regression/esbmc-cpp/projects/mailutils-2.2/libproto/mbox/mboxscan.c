/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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

/* Mailbox Parsing. */

/* Credits to the c-client and its Authors
 * The notorius c-client VALID() macro, was written by Mark Crispin.
 */
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef WITH_PTHREAD
# ifdef HAVE_PTHREAD_H
#  define _XOPEN_SOURCE  500
#  include <pthread.h>
# endif
#endif

#include <stdlib.h>
#include <mbox0.h>

/* Parsing.
   The approach is to detect the "From " as start of a new message, give the
   position of the header and scan until "\n" then set header_end, set body
   position, scan until we it another "From " and set body_end.
   ************************************                                       
   This is a classic case of premature optimisation being the root of all
   Evil(Donald E. Knuth).  But I'm under "pressure" ;-) to come with
   something "faster".  I think it's wastefull * to spend time to gain a few
   seconds on 30Megs mailboxes ... but then again ... in computer time, 60
   seconds, is eternity.  If they use the event notification stuff to get
   some headers/messages early ... it's like pissing in the wind(sorry don't
   have the english equivalent).  The worst is progress_bar it should be ...
   &*($^ nuke.  For the events, we have to remove the *.LCK file, release the
   locks, flush the stream save the pointers  etc ... hurry and wait...
   I this point I'm pretty much ranting.  */

/* You are not expected to understand this macro, but read the next page if
 * you are not faint of heart.
 *  
 * Known formats to the VALID macro are:
 *              From user Wed Dec  2 05:53 1992
 * BSD          From user Wed Dec  2 05:53:22 1992
 * SysV         From user Wed Dec  2 05:53 PST 1992
 * rn           From user Wed Dec  2 05:53:22 PST 1992
 *              From user Wed Dec  2 05:53 -0700 1992
 *              From user Wed Dec  2 05:53:22 -0700 1992
 *              From user Wed Dec  2 05:53 1992 PST
 *              From user Wed Dec  2 05:53:22 1992 PST
 *              From user Wed Dec  2 05:53 1992 -0700
 * Solaris      From user Wed Dec  2 05:53:22 1992 -0700
 *  
 * Plus all of the above with `` remote from xxx'' after it. Thank you very
 * much, smail and Solaris, for making my life considerably more complicated.
 */
/*
 * What?  You want to understand the VALID macro anyway?  Alright, since you
 * insist.  Actually, it isn't really all that difficult, provided that you
 * take it step by step.
 *  
 * Line 1       Initializes the return ti value to failure (0);
 * Lines 2-3    Validates that the 1st-5th characters are ``From ''.
 * Lines 4-6    Validates that there is an end of line and points x at it.
 * Lines 7-14   First checks to see if the line is at least 41 characters long
.
 *              If so, it scans backwards to find the rightmost space.  From
 *              that point, it scans backwards to see if the string matches
 *              `` remote from''.  If so, it sets x to point to the space at
 *              the start of the string.
 * Line 15      Makes sure that there are at least 27 characters in the line.
 * Lines 16-21  Checks if the date/time ends with the year (there is a space
 *              five characters back).  If there is a colon three characters
 *              further back, there is no timezone field, so zn is set to 0
 *              and ti is set in front of the year.  Otherwise, there must
 *              either to be a space four characters back for a three-letter
 *              timezone, or a space six characters back followed by a + or -
 *              for a numeric timezone; in either case, zn and ti become the
 *              offset of the space immediately before it.
 * Lines 22-24  Are the failure case for line 14.  If there is a space four
 *              characters back, it is a three-letter timezone; there must be
a
 *              space for the year nine characters back.  zn is the zone
 *              offset; ti is the offset of the space.
 * Lines 25-28  Are the failure case for line 20.  If there is a space six
 *              characters back, it is a numeric timezone; there must be a
 *              space eleven characters back and a + or - five characters back
.
 *              zn is the zone offset; ti is the offset of the space.
 * Line 29-32   If ti is valid, make sure that the string before ti is of the
 *              form www mmm dd hh:mm or www mmm dd hh:mm:ss, otherwise
 *              invalidate ti.  There must be a colon three characters back
 *              and a space six or nine characters back (depending upon
 *              whether or not the character six characters back is a colon).
 *              There must be a space three characters further back (in front
 *              of the day), one seven characters back (in front of the month)
,
 *              and one eleven characters back (in front of the day of week).
 *              ti is set to be the offset of the space before the time.
 *  
 * Why a macro?  It gets invoked a *lot* in a tight loop.  On some of the
 * newer pipelined machines it is faster being open-coded than it would be if
 * subroutines are called.
 *  
 * Why does it scan backwards from the end of the line, instead of doing the
 * much easier forward scan?  There is no deterministic way to parse the
 * ``user'' field, because it may contain unquoted spaces!  Yes, I tested it t
o
 * see if unquoted spaces were possible.  They are, and I've encountered enoug
h
 * evil mail to be totally unwilling to trust that ``it will never happen''.
 */
#define VALID(s,x,ti,zn) do					              \
  {									      \
    ti = 0;								      \
    if ((*s == 'F') && (s[1] == 'r') && (s[2] == 'o') && (s[3] == 'm') &&     \
	(s[4] == ' '))							      \
      {									      \
	for (x = s + 5; *x && *x != '\n'; x++);				      \
	if (x)								      \
	  {								      \
	    if (x - s >= 41)						      \
	      {								      \
		for (zn = -1; x[zn] != ' '; zn--);			      \
		if ((x[zn - 1] == 'm') && (x[zn - 2] == 'o')		      \
		    && (x[zn - 3] == 'r') && (x[zn - 4] == 'f')		      \
		    && (x[zn - 5] == ' ') && (x[zn - 6] == 'e')		      \
		    && (x[zn - 7] == 't') && (x[zn - 8] == 'o')		      \
		    && (x[zn - 9] == 'm') && (x[zn - 10] == 'e')	      \
		    && (x[zn - 11] == 'r') && (x[zn - 12] == ' '))	      \
		  x += zn - 12;						      \
	      }								      \
	    if (x - s >= 27)						      \
	      {								      \
		if (x[-5] == ' ')					      \
		  {							      \
		    if (x[-8] == ':')					      \
		      zn = 0, ti = -5;					      \
		    else if (x[-9] == ' ')				      \
		      ti = zn = -9;					      \
		    else if ((x[-11] == ' ')				      \
			     && ((x[-10] == '+') || (x[-10] == '-')))	      \
		      ti = zn = -11;					      \
		  }							      \
		else if (x[-4] == ' ')					      \
		  {							      \
		    if (x[-9] == ' ')					      \
		      zn = -4, ti = -9;					      \
		  }							      \
		else if (x[-6] == ' ')					      \
		  {							      \
		    if ((x[-11] == ' ') && ((x[-5] == '+') || (x[-5] == '-')))\
		      zn = -6, ti = -11;				      \
		  }							      \
		if (ti && !((x[ti - 3] == ':') &&			      \
			    (x[ti -= ((x[ti - 6] == ':') ? 9 : 6)] == ' ') && \
			    (x[ti - 3] == ' ') && (x[ti - 7] == ' ') &&	      \
			    (x[ti - 11] == ' ')))			      \
		  ti = 0;						      \
	      }								      \
	  }								      \
      }									      \
  }									      \
while (0)								      
     

#define ATTRIBUTE_SET(buf,mum,c0,c1,type)                                     \
do                                                                            \
{                                                                             \
  char *s;                                                                    \
  for (s = (buf) + 7; *s; s++)                                                \
  {                                                                           \
    if (*s == c0 || *s == c1)                                                 \
      {                                                                       \
        (mum)->attr_flags |= (type);                                          \
        break;                                                                \
      }                                                                       \
  }                                                                           \
} while (0)


/* Notifications ADD_MESG. */						      
#define DISPATCH_ADD_MSG(mbox,mud)                                            \
do                                                                            \
{                                                                             \
  int bailing = 0;                                                            \
  mu_monitor_unlock (mbox->monitor);                                          \
  if (mbox->observable)                                                       \
    {                                                                         \
      size_t tmp = mud->messages_count + 1;                                   \
      bailing = mu_observable_notify (mbox->observable, MU_EVT_MESSAGE_ADD,   \
                                      &tmp);                                  \
    }                                                                         \
  if (bailing != 0)                                                           \
    {                                                                         \
      mu_locker_unlock (mbox->locker);                                        \
      return EINTR;                                                           \
    }                                                                         \
  mu_monitor_wrlock (mbox->monitor);                                          \
} while (0);

/* Notification MBX_PROGRESS
   We do not want to fire up the progress notification every line, it will be
   too expensive, so we do it arbitrarely every 10 000 Lines.
   FIXME: maybe this should be configurable.  */
/* This is more tricky we can not leave the mum struct incomplete.  So we
   only tell them about the complete messages.  */
#define DISPATCH_PROGRESS(mbox,mud)                                          \
do                                                                           \
{                                                                            \
  int bailing = 0;                                                           \
  mu_monitor_unlock (mbox->monitor);                                         \
  mud->messages_count--;                                                     \
  if (mbox->observable)                                                      \
    bailing = mu_observable_notify (mbox->observable,                        \
                                    MU_EVT_MAILBOX_PROGRESS, NULL);          \
  if (bailing != 0)                                                          \
    {	                                                                     \
       mu_locker_unlock (mbox->locker);                                      \
       return EINTR;                                                         \
    }                                                                        \
  mud->messages_count++;                                                     \
  mu_monitor_wrlock (mbox->monitor);                                         \
} while (0)

/* Allocate slots for the new messages.  */
/*    size_t num = 2 * ((mud)->messages_count) + 10; */
#define ALLOCATE_MSGS(mbox,mud)                                              \
do                                                                           \
{                                                                            \
  if ((mud)->messages_count >= (mud)->umessages_count)                       \
    {                                                                        \
      mbox_message_t *m;                                                     \
      size_t num = ((mud)->umessages_count) + 1;                             \
      m = realloc ((mud)->umessages, num * sizeof (*m));                     \
    if (m == NULL)                                                           \
      {                                                                      \
        mu_locker_unlock (mbox->locker);                                     \
        mu_monitor_unlock (mbox->monitor);                                   \
        return ENOMEM;                                                       \
      }                                                                      \
    (mud)->umessages = m;                                                    \
    (mud)->umessages[num - 1] = calloc (1, sizeof (*(mum)));                 \
    if ((mud)->umessages[num - 1] == NULL)                                   \
      {                                                                      \
        mu_locker_unlock (mbox->locker);                                     \
        mu_monitor_unlock (mbox->monitor);                                   \
        return ENOMEM;                                                       \
      }                                                                      \
    (mud)->umessages_count = num;                                            \
  }                                                                          \
} while (0)

#define ISSTATUS(buf) (                                                       \
(buf[0] == 'S' || buf[0] == 's')                                              \
 && (buf[1] == 'T' || buf[1] == 't')                                          \
 && (buf[2] == 'A' || buf[2] == 'a')                                          \
 && (buf[3] == 'T' || buf[3] == 't')                                          \
 && (buf[4] == 'U' || buf[4] == 'u')                                          \
 && (buf[5] == 'S' || buf[5] == 's')                                          \
 && (buf[6] == ':' || buf[6] == ' ' || buf[6] == '\t'))

#define MBOX_SCAN_NOTIFY 0x1
#define MBOX_SCAN_ONEMSG 0x2

int
mbox_scan_internal (mu_mailbox_t mailbox, mbox_message_t mum,
		    mu_off_t total,
		    size_t *pmin_uid,
		    int flags)
{
#define MSGLINELEN 1024
  char buf[MSGLINELEN];
  int inheader;
  int inbody;
  mbox_data_t mud = mailbox->data;
  int status = 0;
  size_t lines;
  int newline;
  size_t n = 0;
  mu_stream_t stream;
  size_t min_uid = 0;
  int zn, isfrom = 0;
  char *temp;
  
  newline = 1;
  errno = lines = inheader = inbody = 0;

  stream = mailbox->stream;
  while ((status = mu_stream_readline (stream, buf, sizeof (buf),
				       total, &n)) == 0 && n != 0)
    {
      int nl;
      total += n;

      nl = (*buf == '\n') ? 1 : 0;
      VALID (buf, temp, isfrom, zn);
      isfrom = (isfrom) ? 1 : 0;

      if ((flags & MBOX_SCAN_ONEMSG) && mum == NULL)
	{
	  /* In one-message mode, the positioning should be exact. */
	  if (!isfrom)
	    return EINVAL; /* FIXME: Better error code, please? */
	}
      
      /* Which part of the message are we in ?  */
      inheader = isfrom | ((!nl) & inheader);
      inbody = (!isfrom) & (!inheader);

      if (buf[n - 1] == '\n')
	lines++;

      if (inheader)
	{
	  /* New message.  */
	  if (isfrom)
	    {
	      /* Signal the end of the body.  */
	      if (mum && !mum->body_end)
		{
		  mum->body_end = total - n - newline;
		  mum->body_lines = --lines - newline;

		  if (mum->uid <= min_uid)
		    {
		      mum->uid = ++min_uid;
		      /* Note that modification for when expunging.  */
		      mum->attr_flags |= MU_ATTRIBUTE_MODIFIED;
		    }
		  else
		    min_uid = mum->uid;

		  if (flags & MBOX_SCAN_ONEMSG)
		    break;
		  
		  if (flags & MBOX_SCAN_NOTIFY)
		    DISPATCH_ADD_MSG (mailbox, mud);
		}
	      /* Allocate_msgs will initialize mum.  */
	      ALLOCATE_MSGS (mailbox, mud);
	      mud->messages_count++;
	      mum = mud->umessages[mud->messages_count - 1];
	      mum->mud = mud;
              mum->header_from = total - n;
              mum->header_from_end = total;
	      mum->body_end = mum->body = 0;
	      mum->attr_flags = 0;
	      lines = 0;
	    }
	  else if (ISSTATUS(buf))
	    {
	      ATTRIBUTE_SET(buf, mum, 'r', 'R', MU_ATTRIBUTE_READ);
	      ATTRIBUTE_SET(buf, mum, 'o', 'O', MU_ATTRIBUTE_SEEN);
	      ATTRIBUTE_SET(buf, mum, 'a', 'A', MU_ATTRIBUTE_ANSWERED);
	      ATTRIBUTE_SET(buf, mum, 'd', 'D', MU_ATTRIBUTE_DELETED);
	    }
	}

      /* Body.  */
      if (inbody)
	{
	  /* Set the body position.  */
	  if (mum && !mum->body)
	    {
	      mum->body = total - n + nl;
	      mum->header_lines = lines;
	      lines = 0;
	    }
	}

      newline = (inbody && lines) ? nl : 0;
      
      /* Every 100 mesgs update the lock, it should be every minute.  */
      if ((mud->messages_count % 100) == 0)
	mu_locker_touchlock (mailbox->locker);

      /* Ping them every 1000 lines. Should be tunable.  */
      if (flags & MBOX_SCAN_NOTIFY)
	if (((lines + 1) % 1000) == 0)
	  DISPATCH_PROGRESS (mailbox, mud);

    } /* while */

  if (mum)
    {
      mum->body_end = total - newline;
      mum->body_lines = lines - newline;

      if (mum->uid <= min_uid)
	{
	  mum->uid = ++min_uid;
	  /* Note that modification for when expunging.  */
	  mum->attr_flags |= MU_ATTRIBUTE_MODIFIED;
	}
      else
	min_uid = mum->uid;
      
      if (flags & MBOX_SCAN_NOTIFY)
	DISPATCH_ADD_MSG (mailbox, mud);
    }
  if (pmin_uid)
    *pmin_uid = min_uid;
  return status;
}

int
mbox_scan0 (mu_mailbox_t mailbox, size_t msgno, size_t *pcount, int do_notif)
{
  int status;
  mbox_data_t mud = mailbox->data;
  mbox_message_t mum = NULL;
  mu_off_t total = 0;
  size_t min_uid;
  
  /* Sanity.  */
  if (mud == NULL)
    return EINVAL;

  /* Grab the lock.  */
  mu_monitor_wrlock (mailbox->monitor);

#ifdef WITH_PTHREAD
  /* read() is cancellation point since we're doing a potentially
     long operation.  Lets make sure we clean the state.  */
  pthread_cleanup_push (mbox_cleanup, (void *)mailbox);
#endif

  /* Save the timestamp and size.  */
  status = mu_stream_size (mailbox->stream, &mud->size);
  if (status != 0)
    {
      mu_monitor_unlock (mailbox->monitor);
      return status;
    }

  if ((status = mu_locker_lock (mailbox->locker)))
    {
      mu_monitor_unlock (mailbox->monitor);
      return status;
    }

  /* Seek to the starting point.  */
  if (mud->umessages && msgno > 0 && mud->messages_count > 0
      && msgno <= mud->messages_count)
    {
      mum = mud->umessages[msgno - 1];
      if (mum)
	total = mum->header_from;
      mud->messages_count = msgno - 1;
    }
  else
    mud->messages_count = 0;

  status = mbox_scan_internal (mailbox, mum, total, &min_uid,
			       do_notif ? MBOX_SCAN_NOTIFY : 0);
    
  if (pcount)
    *pcount = mud->messages_count;
  mu_locker_unlock (mailbox->locker);
  mu_monitor_unlock (mailbox->monitor);

  /* Reset the uidvalidity.  */
  if (mud->messages_count > 0)
    {
      mum = mud->umessages[0];
      if (mud->uidvalidity == 0)
	{
	  mud->uidvalidity = (unsigned long)time (NULL);
	  mud->uidnext = mud->messages_count + 1;
	  /* Tell that we have been modified for expunging.  */
	  mum->attr_flags |= MU_ATTRIBUTE_MODIFIED;
	}
    }
      
  if (mud->messages_count > 0 && min_uid >= mud->uidnext)
    {
      mum = mud->umessages[0];
      mud->uidnext = min_uid + 1;
      mum->attr_flags |= MU_ATTRIBUTE_MODIFIED;
    }

#ifdef WITH_PTHREAD
  pthread_cleanup_pop (0);
#endif

  return status;
}

int
mbox_scan1 (mu_mailbox_t mailbox, mu_off_t offset, int do_notif)
{
  int status;
  mbox_data_t mud = mailbox->data;

  if (mud == NULL)
    return EINVAL;

  /* Grab the lock.  */
  mu_monitor_wrlock (mailbox->monitor);

#ifdef WITH_PTHREAD
  /* read() is cancellation point since we're doing a potentially
     long operation.  Lets make sure we clean the state.  */
  pthread_cleanup_push (mbox_cleanup, (void *)mailbox);
#endif

  if ((status = mu_locker_lock (mailbox->locker)))
    {
      mu_monitor_unlock (mailbox->monitor);
      return status;
    }

  status = mu_stream_seek (mailbox->stream, offset, SEEK_SET);
  if (status)
    {
      mu_monitor_unlock (mailbox->monitor);
      mu_locker_unlock (mailbox->locker);
      return status;
    }

  status = mbox_scan_internal (mailbox, NULL, offset, NULL,
			       MBOX_SCAN_ONEMSG |
			       (do_notif ? MBOX_SCAN_NOTIFY : 0));

  mu_locker_unlock (mailbox->locker);
  mu_monitor_unlock (mailbox->monitor);
  
#ifdef WITH_PTHREAD
  pthread_cleanup_pop (0);
#endif

  return status;
}
