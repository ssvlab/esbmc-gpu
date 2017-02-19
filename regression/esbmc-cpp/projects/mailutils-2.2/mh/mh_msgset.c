/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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

/* MH message sets. */

#include <mh.h>
#include <mailutils/argcv.h>

/* Expand a message set (msgcnt;msglist) to accomodate `inc' more
   elements */
static void
_expand (size_t *msgcnt, size_t **msglist, size_t inc)
{
  if (!inc)
    return;

  *msgcnt += inc;
  *msglist = realloc (*msglist, (*msgcnt)*sizeof(**msglist));
  if (!*msglist)
    mh_err_memory (1);
}

/* Fatal error handler */ 
static void
msgset_abort (const char *arg)
{
  mu_error (_("bad message list `%s'"), arg);
  exit (1);
}

/* Handlers for expansion of the reserved message names */

static int
msgset_first (mu_mailbox_t mbox, size_t *pnum)
{
  *pnum = 1;
  return 0;
}

static int
msgset_last (mu_mailbox_t mbox, size_t *pnum)
{
  int rc;
  size_t count = 0;

  rc = mu_mailbox_messages_count (mbox, &count);
  if (rc)
    {
      mu_error (_("cannot get last message: %s"), mu_strerror (rc));
      exit (1);
    }
  *pnum = count;
  return 0;
}

static int
msgset_cur (mu_mailbox_t mbox, size_t *pnum)
{
  size_t i, count = 0;
  static int cached_n = 0;

  if (cached_n)
    {
      *pnum = cached_n;
      return 0;
    }
  
  mu_mailbox_messages_count (mbox, &count);
  for (i = 1; i <= count; i++)
    {
      mu_message_t msg = NULL;
      size_t uid = 0;
      
      mu_mailbox_get_message (mbox, i, &msg);
      mh_message_number (msg, &uid);
      if (uid == current_message)
	{
	  *pnum = cached_n = i;
	  return 0;
	}
    }
  mu_error (_("no cur message"));
  exit (1);
}

static int
msgset_prev (mu_mailbox_t mbox, size_t *pnum)
{
  size_t cur_n = 0;
  msgset_cur (mbox, &cur_n);
  if (cur_n < 1)
    {
      mu_error (_("no prev message"));
      exit (1);
    }
  *pnum = cur_n - 1;
  return 0;
}

static int
msgset_next (mu_mailbox_t mbox, size_t *pnum)
{
  size_t cur_n = 0, total = 0;
  msgset_cur (mbox, &cur_n);
  mu_mailbox_messages_count (mbox, &total);
  if (cur_n + 1 > total)
    {
      mu_error (_("no next message"));
      exit (1);
    }
  *pnum = cur_n + 1;
  return 0;
}

static struct msgset_keyword {
  char *name;
  int (*handler) (mu_mailbox_t mbox, size_t *pnum);
} keywords[] = {
  { "first", msgset_first },
  { "last", msgset_last },
  { "prev", msgset_prev },
  { "next", msgset_next },
  { "cur", msgset_cur },
  { NULL },
};

/* Preprocess a part of a complex message designation. Returns
   a pointer to the allocated memory containing expanded part of
   the designation. Pointer to the beginning of the not expanded
   part (in arg) is placed into *rest */
static char *
msgset_preproc_part (mu_mailbox_t mbox, char *arg, char **rest)
{
  struct msgset_keyword *p;
  char *cp;
  
  for (p = keywords; p->name; p++)
    if (strncmp (arg, p->name, strlen (p->name)) == 0)
      {
	int rc;
	size_t uid, num;
	mu_message_t msg;
	
	if (p->handler (mbox, &num))
	  msgset_abort (arg);
	rc = mu_mailbox_get_message (mbox, num, &msg);
	if (rc)
	  {
	    mu_error (_("cannot get message %lu: %s"),
		      (unsigned long) num, mu_strerror (rc));
	    exit (1);
	  }
	*rest = arg + strlen (p->name);
	mu_message_get_uid (msg, &uid);
	return xstrdup (mu_umaxtostr (0, uid));
      }
  cp = strchr (arg, '-');
  if (cp)
    {
      char *ret;

      *rest = cp;
      ret = xmalloc (cp - arg + 1);
      memcpy (ret, arg, cp - arg);
      ret[cp - arg] = 0;
      return ret;
    }

  *rest = arg + strlen (arg);
  return strdup (arg);
}

/* Preprocess (expand) a single message designation */
static char *
msgset_preproc (mu_mailbox_t mbox, char *arg)
{
  char *buf, *tail;
  
  if (strcmp (arg, "all") == 0 || strcmp (arg, ".") == 0)
    {
      /* Special case */
      arg = "first-last";
    }

  buf = msgset_preproc_part (mbox, arg, &tail);
  if (tail[0] == '-')
    {
      char *rest = msgset_preproc_part (mbox, tail+1, &tail);
      char *p = NULL;
      asprintf (&p, "%s-%s", buf, rest);
      free (rest);
      free (buf);
      buf = p;
    }
  
  if (tail[0])
    {
      char *p = NULL;
      asprintf (&p, "%s%s", buf, tail);
      free (buf);
      buf = p;
    }
  return buf;
}

static int
comp_mesg (const void *a, const void *b)
{
  if (*(size_t*)a > *(size_t*)b)
    return 1;
  else if (*(size_t*)a < *(size_t*)b)
    return -1;
  return 0;
}

static int _mh_msgset_parse (mu_mailbox_t mbox, mh_msgset_t *msgset,
			     int argc, char **argv);

/* Treat arg as a name of user-defined sequence and attempt to
   expand it. Return 0 if succeeded, non-zero otherwise. */
int
expand_user_seq (mu_mailbox_t mbox, mh_msgset_t *msgset, char *arg)
{
  int argc;
  char **argv;
  char *p;
  const char *listp;
  int rc = 1;
  int negate = 0;
  
  p = strchr (arg, ':');
  if (p)
    *p++ = 0;
  listp = mh_global_sequences_get (arg, NULL);
  if (!listp)
    {
      int len;
      const char *neg = mh_global_profile_get ("Sequence-Negation", NULL);
      if (!neg)
	return 1;
      len = strlen (neg);
      if (strncmp (arg, neg, len))
	return 1;
      negate = 1;
      listp = mh_global_sequences_get (arg + len, NULL);
      if (!listp)
	return 1;
    }
  
  if (mu_argcv_get (listp, "", NULL, &argc, &argv) == 0)
    rc = _mh_msgset_parse (mbox, msgset, argc, argv);
  mu_argcv_free (argc, argv);
  if (rc)
    return rc;

  if (negate)
    mh_msgset_negate (mbox, msgset);
  
  if (p)
    {
      int first, num;
      
      num = strtoul (p, &p, 0);
      if (*p)
	{
	  mh_msgset_free (msgset);
	  return 1;
	}
      if (num < 0)
	{
	  first = num + msgset->count;
	  num = - num;
	}
      else
	first = 0;
      if (num > msgset->count)
	{
	  mh_msgset_free (msgset);
	  return 1;
	}

      if (first > 0)
	memmove (msgset->list, &msgset->list[first],
		 sizeof (msgset->list[0]) * num);
      msgset->count = num;
    }
  
  return rc;
}

/* Parse a message specification from (argc;argv). Returned msgset is
   not sorted nor optimised */
int
_mh_msgset_parse (mu_mailbox_t mbox, mh_msgset_t *msgset, int argc, char **argv)
{
  size_t msgcnt;
  size_t *msglist;
  size_t i, msgno;
  
  if (argc == 0)
    return 1;
  
  msgcnt = argc;
  msglist = calloc (msgcnt, sizeof(*msglist));
  for (i = 0, msgno = 0; i < argc; i++)
    {
      char *p = NULL, *q;
      size_t start, end;
      size_t msg_first, n;
      long num;
      char *arg = msgset_preproc (mbox, argv[i]);

      if (!mu_isdigit (arg[0]))
	{
	  int j;
	  mh_msgset_t m;
	  
	  if (expand_user_seq (mbox, &m, arg))
	    {
	      mu_error (_("message set %s does not exist"), arg);
	      exit (1);
	    }
	  _expand (&msgcnt, &msglist, m.count);
	  for (j = 0; j < m.count; j++)
	    msglist[msgno++] = m.list[j];
	  mh_msgset_free (&m);
	}
      else
	{
	  start = strtoul (arg, &p, 0);
	  switch (*p)
	    {
	    case 0:
	      n = mh_get_message (mbox, start, NULL);
	      if (!n)
		{
		  mu_error (_("message %lu does not exist"),
			    (unsigned long) start);
		  exit (1);
		}
	      msglist[msgno++] = n;
	      break;
	      
	    case '-':
	      end = strtoul (p+1, &p, 0);
	      if (*p)
		msgset_abort (argv[i]);
	      if (end < start)
		{
		  size_t t = start;
		  start = end;
		  end = t;
		}
	      _expand (&msgcnt, &msglist, end - start);
	      msg_first  = msgno;
	      for (; start <= end; start++)
		{
		  n = mh_get_message (mbox, start, NULL);
		  if (n)
		    msglist[msgno++] = n;
		}
	      if (msgno == msg_first)
		{
		  mu_error (_("no messages in range %s"), argv[i]);
		  exit (1);
		}
	      break;
	      
	    case ':':
	      num = strtoul (p+1, &q, 0);
	      if (*q)
		msgset_abort (argv[i]);
	      if (p[1] != '+' && p[1] != '-')
		{
		  if (strncmp (argv[i], "last:", 5) == 0
		      || strncmp (argv[i], "prev:", 5) == 0)
		    num = -num;
		}
	      end = start + num;
	      if (end < start)
		{
		  size_t t = start;
		  start = end + 1;
		  end = t;
		}
	      else
		end--;
	      _expand (&msgcnt, &msglist, end - start);
	      msg_first  = msgno;
	      for (; start <= end; start++)
		{
		  n = mh_get_message (mbox, start, NULL);
		  if (n)
		    msglist[msgno++] = n;
		}
	      if (msgno == msg_first)
		{
		  mu_error (_("no messages in range %s"), argv[i]);
		  exit (1);
		}
	      break;
	      
	    default:
	      msgset_abort (argv[i]);
	    }
	}
      free (arg);
    }

  msgset->count = msgno;
  msgset->list = msglist;
  return 0;
}

/* Parse a message specification from (argc;argv). Returned msgset is
   sorted and optimised (i.e. it does not contain duplicate message
   numbers) */
int
mh_msgset_parse (mu_mailbox_t mbox, mh_msgset_t *msgset,
		 int argc, char **argv, char *def)
{
  char *xargv[2];
  int rc;
  
  if (argc == 0)
    {
      argc = 1;
      argv = xargv;
      argv[0] = def ? def : "cur";
      argv[1] = NULL;
    }
  
  rc = _mh_msgset_parse (mbox, msgset, argc, argv);

  if (rc == 0)
    {
      size_t i, msgno;
      size_t msgcnt = msgset->count;
      size_t *msglist = msgset->list;
      
      /* Sort the resulting message set */
      qsort (msglist, msgcnt, sizeof (*msgset->list), comp_mesg);

      /* Remove duplicates. */
      for (i = 0, msgno = 1; i < msgset->count; i++)
	if (msglist[msgno-1] != msglist[i])
	  msglist[msgno++] = msglist[i];
      msgset->count = msgno;
    }
  return rc;
}

/* Check if message with ordinal number `num' is contained in the
   message set. */
int
mh_msgset_member (mh_msgset_t *msgset, size_t num)
{
  size_t i;

  for (i = 0; i < msgset->count; i++)
    if (msgset->list[i] == num)
      return i + 1;
  return 0;
}

/* Auxiliary function. Performs binary search for a message with the
   given sequence number */
static size_t
mh_search_message (mu_mailbox_t mbox, size_t start, size_t stop,
		   size_t seqno, mu_message_t *mesg)
{
  mu_message_t mid_msg = NULL;
  size_t num = 0, middle;

  middle = (start + stop) / 2;
  if (mu_mailbox_get_message (mbox, middle, &mid_msg)
      || mh_message_number (mid_msg, &num))
    return 0;

  if (num == seqno)
    {
      if (mesg)
	*mesg = mid_msg;
      return middle;
    }
      
  if (start >= stop)
    return 0;

  if (num > seqno)
    return mh_search_message (mbox, start, middle-1, seqno, mesg);
  else /*if (num < seqno)*/
    return mh_search_message (mbox, middle+1, stop, seqno, mesg);
}

/* Retrieve the message with the given sequence number.
   Returns ordinal number of the message in the mailbox if found,
   zero otherwise. The retrieved message is stored in the location
   pointed to by mesg, unless it is NULL. */
   
size_t
mh_get_message (mu_mailbox_t mbox, size_t seqno, mu_message_t *mesg)
{
  size_t num, count;
  mu_message_t msg;

  if (mu_mailbox_get_message (mbox, 1, &msg)
      || mh_message_number (msg, &num))
    return 0;
  if (seqno < num)
    return 0;
  else if (seqno == num)
    {
      if (mesg)
	*mesg = msg;
      return 1;
    }

  if (mu_mailbox_messages_count (mbox, &count)
      || mu_mailbox_get_message (mbox, count, &msg)
      || mh_message_number (msg, &num))
    return 0;
  if (seqno > num)
    return 0;
  else if (seqno == num)
    {
      if (mesg)
	*mesg = msg;
      return count;
    }

  return mh_search_message (mbox, 1, count, seqno, mesg);
}

/* Reverse the order of messages in the message set */
void
mh_msgset_reverse (mh_msgset_t *msgset)
{
  int head, tail;

  for (head = 0, tail = msgset->count-1; head < tail; head++, tail--)
    {
      size_t val = msgset->list[head];
      msgset->list[head] = msgset->list[tail];
      msgset->list[tail] = val;
    }
}

/* Set the current message to that contained at position `index'
   in the given message set */
int
mh_msgset_current (mu_mailbox_t mbox, mh_msgset_t *msgset, int index)
{
  mu_message_t msg = NULL;
  if (mu_mailbox_get_message (mbox, msgset->list[index], &msg))
    return 1;
  return mh_message_number (msg, &current_message);
}

/* Free memory allocated for the message set. Note, that the msgset
   itself is supposed to reside in the statically allocated memory and
   therefore is not freed */
void
mh_msgset_free (mh_msgset_t *msgset)
{
  if (msgset->count)
    free (msgset->list);
}

/* Negate the message set: on return `msgset' consists of the messages
   _not contained_ in the input message set. Any memory associated with
   the input message set is freed */
void
mh_msgset_negate (mu_mailbox_t mbox, mh_msgset_t *msgset)
{
  size_t i, total = 0, msgno;
  size_t *list;

  mu_mailbox_messages_count (mbox, &total);
  list = calloc (total, sizeof (list[0]));
  if (!list)
    mh_err_memory (1);
  for (i = 1, msgno = 0; i <= total; i++)
    {
      if (!mh_msgset_member (msgset, i))
	list[msgno++] = i;
    }

  list = realloc (list, sizeof (list[0]) * msgno);
  if (!list)
    {
      mu_error (_("not enough memory"));
      abort ();
    }
  mh_msgset_free (msgset);
  msgset->count = msgno;
  msgset->list = list;
}

void
mh_msgset_uids (mu_mailbox_t mbox, mh_msgset_t *msgset)
{
  size_t i;
  for (i = 0; i < msgset->count; i++)
    {
      mu_message_t msg;
      mu_mailbox_get_message (mbox, msgset->list[i], &msg);
      mh_message_number (msg, &msgset->list[i]);
    }
}
