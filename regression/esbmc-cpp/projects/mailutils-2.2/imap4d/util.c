/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
   2009, 2010 Free Software Foundation, Inc.

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

static mu_stream_t istream;
static mu_stream_t ostream;

static int add2set (size_t **, int *, unsigned long);
static const char *sc2string (int);

/* NOTE: Allocates Memory.  */
/* Expand: ~ --> /home/user and to ~guest --> /home/guest.  */
char *
util_tilde_expansion (const char *ref, const char *delim)
{
  return mu_tilde_expansion (ref, delim, imap4d_homedir);
}

/* Get the absolute path.  */
/* NOTE: Path is allocated and must be free()d by the caller.  */
char *
util_getfullpath (const char *name, const char *delim)
{
  char *p = util_tilde_expansion (name, delim);
  if (*p != delim[0])
    {
      char *s =
	calloc (strlen (imap4d_homedir) + strlen (delim) + strlen (p) + 1, 1);
      sprintf (s, "%s%s%s", imap4d_homedir, delim, p);
      free (p);
      p = s;
    }
  return mu_normalize_path (p);
}

static int
comp_int (const void *a, const void *b)
{
  return *(int *) a - *(int *) b;
}

/* Parse the message set specification from S. Store message numbers
   in SET, store number of element in the SET into the memory pointed to
   by N.

   A message set is defined as:

   set ::= sequence_num / (sequence_num ":" sequence_num) / (set "," set)
   sequence_num    ::= nz_number / "*"
   ;; * is the largest number in use.  For message
   ;; sequence numbers, it is the number of messages
   ;; in the mailbox.  For unique identifiers, it is
   ;; the unique identifier of the last message in
   ;; the mailbox.
   nz_number       ::= digit_nz *digit

   FIXME: The message sets like <,,,> or <:12> or <20:10> are not considered
   an error */
int
util_msgset (char *s, size_t ** set, int *n, int isuid)
{
  unsigned long val = 0;
  unsigned long low = 0;
  int done = 0;
  int status = 0;
  size_t max = 0;
  size_t *tmp;
  int i, j;
  unsigned long invalid_uid = 0; /* For UID mode only: have we 
				    encountered an uid > max uid? */
  
  status = mu_mailbox_messages_count (mbox, &max);
  if (status != 0)
    return status;
  /* If it is a uid sequence, override max with the UID.  */
  if (isuid)
    {
      mu_message_t msg = NULL;
      mu_mailbox_get_message (mbox, max, &msg);
      mu_message_get_uid (msg, &max);
    }

  *n = 0;
  *set = NULL;
  while (*s)
    {
      switch (*s)
	{
	  /* isdigit */
	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
	  {
	    errno = 0;
	    val = strtoul (s, &s, 10);
	    if (val == ULONG_MAX && errno == ERANGE)
	      {
		if (*set)
		  free (*set);
		*set = NULL;
		*n = 0;
		return EINVAL;
	      }
	    else if (val > max)
	      {
		if (isuid)
		  {
		    invalid_uid = 1;
		    continue;
		  }
		if (*set)
		  free (*set);
		*set = NULL;
		*n = 0;
		return EINVAL;
	      }
	    
	    if (low)
	      {
		/* Reverse it. */
		if (low > val)
		  {
		    long tmp = low;
		    tmp -= 2;
		    if (tmp < 0 || val == 0)
		      {
			free (*set);
			*set = NULL;
			*n = 0;
			return EINVAL;
		      }
		    low = val;
		    val = tmp;
		  }
		for (; low && low <= val; low++)
		  {
		    status = add2set (set, n, low);
		    if (status != 0)
		      return status;
		  }
		low = 0;
	      }
	    else
	      {
		status = add2set (set, n, val);
		if (status != 0)
		  return status;
	      }
	    break;
	  }

	  /* A pair of numbers separated by a ':' character indicates a
	     contiguous set of mesages ranging from the first number to the
	     second:
	     3:5  --> 3 4 5
	   */
	case ':':
	  low = val + 1;
	  s++;
	  break;

	  /* As a convenience. '*' is provided to refer to the highest
	     message number int the mailbox:
	     5:*  --> 5 6 7 8
	   */
	case '*':
	  {
	    val = max;
	    s++;
	    status = add2set (set, n, val);
	    if (status != 0)
	      return status;
	  }
	  break;

	  /* IMAP also allows a set of noncontiguous numbers to be specified
	     with the ',' character:
	     1,3,5,7  --> 1 3 5 7
	   */
	case ',':
	  s++;
	  break;

	default:
	  done = 1;
	  if (*set)
	    free (*set);
	  *set = NULL;
	  *n = 0;
	  return EINVAL;

	}			/* switch */

      if (done)
	break;
    }				/* while */

  if (*n == 0)
    return 0;
  
  /* For message sets in form X:Y where Y is a not-existing UID greater
     than max UID, replace Y with the max UID in the mailbox */
  if (*n == 1 && invalid_uid)
    {
      val = max;
      status = add2set (set, n, val);
      if (status != 0)
	return status;
    }
  
  if (low)
    {
      /* Reverse it. */
      if (low > val)
	{
	  long tmp = low;
	  tmp -= 2;
	  if (tmp < 0 || val == 0)
	    {
	      free (*set);
	      *set = NULL;
	      *n = 0;
	      return EINVAL;
	    }
	  low = val;
	  val = tmp;
	}
      for (; low && low <= val; low++)
	{
	  status = add2set (set, n, low);
	  if (status != 0)
	    return status;
	}
    }

  /* Sort the resulting message set */
  qsort (*set, *n, sizeof (**set), comp_int);

  /* Remove duplicates. tmp serves to avoid extra dereferences */
  tmp = *set;
  for (i = 0, j = 1; i < *n; i++)
    if (tmp[j - 1] != tmp[i])
      tmp[j++] = tmp[i];
  *n = j;
  return 0;
}

int
util_send_bytes (const char *buf, size_t size)
{
  return mu_stream_sequential_write (ostream, buf, size);
}

int
util_send (const char *format, ...)
{
  char *buf = NULL;
  int status = 0;
  va_list ap;

  va_start (ap, format);
  vasprintf (&buf, format, ap);
  va_end (ap);
  if (!buf)
      imap4d_bye (ERR_NO_MEM);

#if 0
  if (imap4d_transcript)
    mu_diag_output (MU_DIAG_DEBUG, "sent: %s", buf);
#endif

  status = mu_stream_sequential_write (ostream, buf, strlen (buf));
  free (buf);

  return status;
}

/* Send NIL if empty string, change the quoted string to a literal if the
   string contains: double quotes, CR, LF, and '/'.  CR, LF will be change
   to spaces.  */
int
util_send_qstring (const char *buffer)
{
  if (buffer == NULL || *buffer == '\0')
    return util_send ("NIL");
  if (strchr (buffer, '"') || strchr (buffer, '\r') || strchr (buffer, '\n')
      || strchr (buffer, '\\'))
    {
      char *s;
      int ret;
      char *b = strdup (buffer);
      while ((s = strchr (b, '\n')) || (s = strchr (b, '\r')))
	*s = ' ';
      ret = util_send_literal (b);
      free (b);
      return ret;
    }
  return util_send ("\"%s\"", buffer);
}

int
util_send_literal (const char *buffer)
{
  return util_send ("{%lu}\r\n%s", (unsigned long) strlen (buffer), buffer);
}

/* Send an unsolicited response.  */
int
util_out (int rc, const char *format, ...)
{
  char *tempbuf = NULL;
  char *buf = NULL;
  int status = 0;
  va_list ap;

  asprintf (&tempbuf, "* %s%s\r\n", sc2string (rc), format);
  va_start (ap, format);
  vasprintf (&buf, tempbuf, ap);
  va_end (ap);
  if (!buf)
    imap4d_bye (ERR_NO_MEM);

  if (imap4d_transcript)
    {
      int len = strcspn (buf, "\r\n");
      mu_diag_output (MU_DIAG_DEBUG, "sent: %*.*s", len, len, buf);
    }

  status = mu_stream_sequential_write (ostream, buf, strlen (buf));
  free (buf);
  free (tempbuf);
  return status;
}

/* Send the tag response and reset the state.  */
int
util_finish (struct imap4d_command *command, int rc, const char *format, ...)
{
  size_t size;
  char *buf = NULL;
  char *tempbuf = NULL;
  int new_state;
  int status = 0;
  va_list ap;
  const char *sc = sc2string (rc);
  
  va_start (ap, format);
  vasprintf (&tempbuf, format, ap);
  va_end (ap);
  if (!tempbuf)
    imap4d_bye (ERR_NO_MEM);
  
  size = strlen (command->tag) + 1 +
         strlen (sc) + strlen (command->name) + 1 +
         strlen (tempbuf) + 1;
  buf = malloc (size);
  if (!buf)
    imap4d_bye (ERR_NO_MEM);
  strcpy (buf, command->tag);
  strcat (buf, " ");
  strcat (buf, sc);
  strcat (buf, command->name);
  strcat (buf, " ");
  strcat (buf, tempbuf);
  free (tempbuf);

  if (imap4d_transcript)
    mu_diag_output (MU_DIAG_DEBUG, "sent: %s", buf);

  mu_stream_sequential_write (ostream, buf, strlen (buf));
  free (buf);
  mu_stream_sequential_write (ostream, "\r\n", 2);

  /* Reset the state.  */
  if (rc == RESP_OK)
    new_state = command->success;
  else if (command->failure <= state)
    new_state = command->failure;
  else
    new_state = STATE_NONE;

  if (new_state != STATE_NONE)
    {
      util_run_events (state, new_state);
      state = new_state;
    }
  
  return status;
}

int
util_do_command (imap4d_tokbuf_t tok)
{
  char *tag, *cmd;
  struct imap4d_command *command;
  static struct imap4d_command nullcommand;
  int argc = imap4d_tokbuf_argc (tok);
  
  if (argc == 0)
    {
      nullcommand.name = "";
      nullcommand.tag = (char *) "*";
      return util_finish (&nullcommand, RESP_BAD, "Null command");
    }
  else if (argc == 1)
    {
      nullcommand.name = "";
      nullcommand.tag = imap4d_tokbuf_getarg (tok, 0);
      return util_finish (&nullcommand, RESP_BAD, "Missing command");
    }

  tag = imap4d_tokbuf_getarg (tok, 0);
  cmd = imap4d_tokbuf_getarg (tok, 1);
  
  command = util_getcommand (cmd, imap4d_command_table);
  if (command == NULL)
    {
      nullcommand.name = "";
      nullcommand.tag = tag;
      return util_finish (&nullcommand, RESP_BAD, "Invalid command");
    }

  command->tag = tag;

  if (command->states && (command->states & state) == 0)
    return util_finish (command, RESP_BAD, "Wrong state");

  return command->func (command, tok);
}

struct imap4d_command *
util_getcommand (char *cmd, struct imap4d_command command_table[])
{
  size_t i, len = strlen (cmd);

  for (i = 0; command_table[i].name != 0; i++)
    {
      if (strlen (command_table[i].name) == len &&
	  !mu_c_strcasecmp (command_table[i].name, cmd))
	return &command_table[i];
    }
  return NULL;
}

/* Status Code to String.  */
static const char *
sc2string (int rc)
{
  switch (rc)
    {
    case RESP_OK:
      return "OK ";

    case RESP_BAD:
      return "BAD ";

    case RESP_NO:
      return "NO ";

    case RESP_BYE:
      return "BYE ";

    case RESP_PREAUTH:
      return "PREAUTH ";
    }
  return "";
}

static int
add2set (size_t ** set, int *n, unsigned long val)
{
  size_t *tmp;
  tmp = realloc (*set, (*n + 1) * sizeof (**set));
  if (tmp == NULL)
    {
      if (*set)
	free (*set);
      *n = 0;
      return ENOMEM;
    }
  *set = tmp;
  (*set)[*n] = val;
  (*n)++;
  return 0;
}

int
util_parse_internal_date (char *date, time_t * timep)
{
  struct tm tm;
  mu_timezone tz;
  time_t time;
  char **datep = &date;

  if (mu_parse_imap_date_time ((const char **) datep, &tm, &tz))
    return 1;

  time = mu_tm2time (&tm, &tz);
  if (time == (time_t) - 1)
    return 2;

  *timep = time;
  return 0;
}

int
util_parse_822_date (const char *date, time_t * timep)
{
  struct tm tm;
  mu_timezone tz;
  const char *p = date;

  if (mu_parse822_date_time (&p, date + strlen (date), &tm, &tz) == 0)
    {
      *timep = mu_tm2time (&tm, &tz);
      return 0;
    }
  return 1;
}

int
util_parse_ctime_date (const char *date, time_t * timep)
{
  struct tm tm;
  mu_timezone tz;

  if (mu_parse_ctime_date_time (&date, &tm, &tz) == 0)
    {
      *timep = mu_tm2time (&tm, &tz);
      return 0;
    }
  return 1;
}

/* Return the first ocurrence of NEEDLE in HAYSTACK. Case insensitive
   comparison */
char *
util_strcasestr (const char *haystack, const char *needle)
{
  return mu_strcasestr (haystack, needle);
}

struct
{
  char *name;
  int flag;
}
_imap4d_attrlist[] =
{
  { "\\Answered", MU_ATTRIBUTE_ANSWERED },
  { "\\Flagged", MU_ATTRIBUTE_FLAGGED },
  { "\\Deleted", MU_ATTRIBUTE_DELETED },
  { "\\Draft", MU_ATTRIBUTE_DRAFT },
  { "\\Seen", MU_ATTRIBUTE_READ },
  { "\\Recent", MU_ATTRIBUTE_RECENT },
};

#define NATTR sizeof(_imap4d_attrlist)/sizeof(_imap4d_attrlist[0])

int _imap4d_nattr = NATTR;

int
util_attribute_to_type (const char *item, int *type)
{
  int i;
  for (i = 0; i < _imap4d_nattr; i++)
    if (mu_c_strcasecmp (item, _imap4d_attrlist[i].name) == 0)
      {
	*type = _imap4d_attrlist[i].flag;
	return 0;
      }
  return 1;
}

/* Note: currently unused. Not needed, possibly? */
int
util_type_to_attribute (int type, char **attr_str)
{
  char *attr_list[NATTR];
  int nattr = 0;
  int i;
  size_t len = 0;

  if (MU_ATTRIBUTE_IS_UNSEEN (type))
    *attr_str = strdup ("\\Recent");
  else
    *attr_str = NULL;

  for (i = 0; i < _imap4d_nattr; i++)
    if (type & _imap4d_attrlist[i].flag)
      {
	attr_list[nattr++] = _imap4d_attrlist[i].name;
	len += 1 + strlen (_imap4d_attrlist[i].name);
      }

  *attr_str = malloc (len + 1);
  (*attr_str)[0] = 0;
  if (*attr_str)
    {
      for (i = 0; i < nattr; i++)
	{
	  strcat (*attr_str, attr_list[i]);
	  if (i != nattr - 1)
	    strcat (*attr_str, " ");
	}
    }

  if (!*attr_str)
    imap4d_bye (ERR_NO_MEM);
  return 0;
}

void
util_print_flags (mu_attribute_t attr)
{
  int i;
  int flags = 0;
  int space = 0;

  mu_attribute_get_flags (attr, &flags);
  for (i = 0; i < _imap4d_nattr; i++)
    if (flags & _imap4d_attrlist[i].flag)
      {
	if (space)
	  util_send (" ");
	else
	  space = 1;
	util_send (_imap4d_attrlist[i].name);
      }

  if (MU_ATTRIBUTE_IS_UNSEEN (flags))
    {
      if (space)
	util_send (" ");
      util_send ("\\Recent");
    }
}

int
util_attribute_matches_flag (mu_attribute_t attr, const char *item)
{
  int flags = 0, mask = 0;

  mu_attribute_get_flags (attr, &flags);
  util_attribute_to_type (item, &mask);
  if (mask == MU_ATTRIBUTE_RECENT)
    return MU_ATTRIBUTE_IS_UNSEEN (flags);

  return flags & mask;
}

char *
util_localname ()
{
  static char *localname;

  if (!localname)
    {
      char *name;
      int name_len = 256;
      int status = 1;
      struct hostent *hp;

      name = malloc (name_len);
      while (name
	     && (status = gethostname (name, name_len)) == 0
	     && !memchr (name, 0, name_len))
	{
	  name_len *= 2;
	  name = realloc (name, name_len);
	}
      if (status || name == NULL)
	{
	  mu_diag_output (MU_DIAG_CRIT, _("cannot find out my own hostname"));
	  exit (EX_OSERR);
	}

      hp = gethostbyname (name);
      if (hp)
	{
	  struct in_addr inaddr;
	  inaddr.s_addr = *(unsigned int *) hp->h_addr;
	  hp = gethostbyaddr ((const char *) &inaddr,
			      sizeof (struct in_addr), AF_INET);
	  if (hp)
	    {
	      free (name);
	      name = strdup ((char *) hp->h_name);
	    }
	}
      localname = name;
    }
  return localname;
}

/* Match STRING against the IMAP4 wildcard pattern PATTERN. */

#define WILD_FALSE 0
#define WILD_TRUE  1
#define WILD_ABORT 2

int
_wild_match (const char *expr, const char *name, char delim)
{
  while (expr && *expr)
    {
      if (*name == 0 && *expr != '*')
	return WILD_ABORT;
      switch (*expr)
	{
	case '*':
	  while (*++expr == '*')
	    ;
	  if (*expr == 0)
	    return WILD_TRUE;
	  while (*name)
	    {
	      int res = _wild_match (expr, name++, delim);
	      if (res != WILD_FALSE)
		return res;
	    }
	  return WILD_ABORT;

	case '%':
	  while (*++expr == '%')
	    ;
	  if (*expr == 0)
	    return strchr (name, delim) ? WILD_FALSE : WILD_TRUE;
	  while (*name && *name != delim)
	    {
	      int res = _wild_match (expr, name++, delim);
	      if (res != WILD_FALSE)
		return res;
	    }
	  return _wild_match (expr, name, delim);
	  
	default:
	  if (*expr != *name)
	    return WILD_FALSE;
	  expr++;
	  name++;
	}
    }
  return *name == 0;
}

int
util_wcard_match (const char *name, const char *expr, const char *delim)
{
  return _wild_match (expr, name, delim[0]) != WILD_TRUE;
}

/* Return the uindvalidity of a mailbox.
   When a mailbox is selected, whose first message does not keep X-UIDVALIDITY
   value, the uidvalidity is computed basing on the return of time(). Now,
   if we call "EXAMINE mailbox" or "STATUS mailbox (UIDVALIDITY)" the same
   mailbox is opened second time and the uidvalidity recalculated. Thus each
   subsequent call to EXAMINE or STATUS upon an already selected mailbox
   will return a different uidvalidity value. To avoid this, util_uidvalidity()
   first sees if it is asked to operate upon an already opened mailbox
   and if so, returns the previously computed value. */
int
util_uidvalidity (mu_mailbox_t smbox, unsigned long *uidvp)
{
  mu_url_t mbox_url = NULL;
  mu_url_t smbox_url = NULL;

  mu_mailbox_get_url (mbox, &mbox_url);
  mu_mailbox_get_url (smbox, &smbox_url);
  if (strcmp (mu_url_to_string (mbox_url), mu_url_to_string (smbox_url)) == 0)
    smbox = mbox;
  return mu_mailbox_uidvalidity (smbox, uidvp);
}


void
util_setio (FILE *in, FILE *out)
{
  if (!in)
    imap4d_bye (ERR_NO_IFILE);
  if (!out)
    imap4d_bye (ERR_NO_OFILE);

  setvbuf (in, NULL, _IOLBF, 0);
  setvbuf (out, NULL, _IOLBF, 0);
  if (mu_stdio_stream_create (&istream, in, MU_STREAM_NO_CLOSE))
    imap4d_bye (ERR_NO_IFILE);
  if (mu_stdio_stream_create (&ostream, out, MU_STREAM_NO_CLOSE))
    imap4d_bye (ERR_NO_OFILE);
}

void
util_get_input (mu_stream_t *pstr)
{
  *pstr = istream;
}

void
util_get_output (mu_stream_t *pstr)
{
  *pstr = ostream;
}

void
util_set_input (mu_stream_t str)
{
  istream = str;
}

void
util_set_output (mu_stream_t str)
{
  ostream = str;
}

/* Wait TIMEOUT seconds for data on the input stream.
   Returns 0   if no data available
           1   if some data is available
	   -1  an error occurred */
int
util_wait_input (int timeout)
{
  int wflags = MU_STREAM_READY_RD;
  struct timeval tv;
  int status;
  
  tv.tv_sec = timeout;
  tv.tv_usec = 0;
  status = mu_stream_wait (istream, &wflags, &tv);
  if (status)
    {
      mu_diag_output (MU_DIAG_ERROR, _("cannot poll input stream: %s"),
		      mu_strerror(status));
      return -1;
    }
  return wflags & MU_STREAM_READY_RD;
}

void
util_flush_output ()
{
  mu_stream_flush (ostream);
}

int
util_is_master ()
{
  return ostream == NULL;
}

#ifdef WITH_TLS
int
imap4d_init_tls_server ()
{
  mu_stream_t stream;
  int rc;
 
  rc = mu_tls_stream_create (&stream, istream, ostream, 0);
  if (rc)
    return 0;

  if (mu_stream_open (stream))
    {
      const char *p;
      mu_stream_strerror (stream, &p);
      mu_diag_output (MU_DIAG_ERROR, _("cannot open TLS stream: %s"), p);
      return 0;
    }

  istream = ostream = stream;
  return 1;
}
#endif /* WITH_TLS */

static mu_list_t atexit_list;

void
util_atexit (void (*fp) (void))
{
  if (!atexit_list)
    mu_list_create (&atexit_list);
  mu_list_append (atexit_list, (void*)fp);
}

static int
atexit_run (void *item, void *data)
{
  ((void (*) (void)) item) ();
  return 0;
}

void
util_bye ()
{
  int rc = istream != ostream;
  
  mu_stream_close (istream);
  mu_stream_destroy (&istream, mu_stream_get_owner (istream));

  if (rc)
    {
      mu_stream_close (ostream);
      mu_stream_destroy (&ostream, mu_stream_get_owner (ostream));
    }
      
  mu_list_do (atexit_list, atexit_run, 0);
}

struct state_event {
  int old_state;
  int new_state;
  mu_list_action_t *action;
  void *data;
};

static mu_list_t event_list;

void
util_register_event (int old_state, int new_state,
		     mu_list_action_t *action, void *data)
{
  struct state_event *evp = malloc (sizeof (*evp));
  if (!evp)
    imap4d_bye (ERR_NO_MEM);
  evp->old_state = old_state;
  evp->new_state = new_state;
  evp->action = action;
  evp->data = data;
  if (!event_list)
    {
      mu_list_create (&event_list);
      mu_list_set_destroy_item (event_list, mu_list_free_item);
    }
  mu_list_append (event_list, (void*)evp);
}

void
util_event_remove (void *id)
{
  mu_list_remove (event_list, id);
}

static int
event_exec (void *item, void *data)
{
  struct state_event *ev = data, *elem = item;

  if (ev->old_state == elem->old_state && ev->new_state == elem->new_state)
    return elem->action (item, elem->data);
  return 0;
}

void
util_run_events (int old_state, int new_state)
{
  if (event_list)
    {
      struct state_event ev;
      mu_iterator_t itr;
      ev.old_state = old_state;
      ev.new_state = new_state;

      mu_list_get_iterator (event_list, &itr);
      for (mu_iterator_first (itr);
	   !mu_iterator_is_done (itr); mu_iterator_next (itr))
	{
	  struct state_event *p;
	  mu_iterator_current (itr, (void **)&p);
	  if (event_exec (p, &ev))
	    break;
	}
      mu_iterator_destroy (&itr);
    }
}
  
void
util_chdir (const char *dir)
{
  int rc = chdir (dir);
  if (rc)
    mu_error ("Cannot change to home directory `%s': %s",
	      dir, mu_strerror (errno));
}

int
is_atom (const char *s)
{
  if (strpbrk (s, "(){ \t%*\"\\"))
    return 0;
  for (; *s; s++)
    {
      if (mu_iscntrl (*s))
	return 0;
    }
  return 1;
}
     

static size_t
remove_cr (char *line, size_t len)
{
  char *prev = NULL;
  size_t rlen = len;
  char *p;
  while ((p = memchr (line, '\r', len)))
    {
      if (prev)
	{
	  memmove (prev, line, p - line);
	  prev += p - line;
	}
      else
	prev = p;
      rlen--;
      len -= p - line + 1;
      line = p + 1;
    }
  if (prev)
    memmove (prev, line, len);
  return rlen;
}

static size_t
unquote (char *line, size_t len)
{
  char *prev = NULL;
  size_t rlen = len;
  char *p;
  int off = 0;
  while ((p = memchr (line + off, '\\', len - off)))
    {
      if (p[1] == '\\' || p[1] == '"')
	{
	  if (prev)
	    {
	      memmove (prev, line, p - line);
	      prev += p - line;
	    }
	  else
	    prev = p;
	  off = p[1] == '\\';
	  rlen--;
	  len -= p - line + 1;
	  line = p + 1;
	}
    }
  if (prev)
    memmove (prev, line, len);
  return rlen;
}

struct imap4d_tokbuf {
  char *buffer;
  size_t size;
  size_t level;
  int argc;
  int argmax;
  size_t *argp;
};

struct imap4d_tokbuf *
imap4d_tokbuf_init ()
{
  struct imap4d_tokbuf *tok = malloc (sizeof (tok[0]));
  if (!tok)
    imap4d_bye (ERR_NO_MEM);
  memset (tok, 0, sizeof (*tok));
  return tok;
}

void
imap4d_tokbuf_destroy (struct imap4d_tokbuf **ptok)
{
  struct imap4d_tokbuf *tok = *ptok;
  free (tok->buffer);
  free (tok->argp);
  free (tok);
  *ptok = NULL;
}

int
imap4d_tokbuf_argc (struct imap4d_tokbuf *tok)
{
  return tok->argc;
}

char *
imap4d_tokbuf_getarg (struct imap4d_tokbuf *tok, int n)
{
  if (n < tok->argc)
    return tok->buffer + tok->argp[n];
  return NULL;
}

static void
imap4d_tokbuf_unquote (struct imap4d_tokbuf *tok, size_t *poff, size_t *plen)
{
  char *buf = tok->buffer + *poff;
  if (buf[0] == '"' && buf[*plen - 1] == '"')
    {
      ++*poff;
      *plen = unquote (buf + 1, *plen - 1);
    }
}

static void
imap4d_tokbuf_expand (struct imap4d_tokbuf *tok, size_t size)
{
  if (tok->size - tok->level < size)	       
    {						
      tok->size = tok->level + size;
      tok->buffer = realloc (tok->buffer, tok->size);
      if (!tok->buffer)				
	imap4d_bye (ERR_NO_MEM);
    }
}

#define ISDELIM(c) (strchr ("()", (c)) != NULL)

int
util_isdelim (const char *str)
{
  return str[1] == 0 && ISDELIM (str[0]);
}

static size_t
insert_nul (struct imap4d_tokbuf *tok, size_t off)
{
  imap4d_tokbuf_expand (tok, 1);
  if (off < tok->level)
    {
      memmove (tok->buffer + off + 1, tok->buffer + off, tok->level - off);
      tok->level++;
    }
  tok->buffer[off] = 0;
  return off + 1;
}

static size_t
gettok (struct imap4d_tokbuf *tok, size_t off)
{
  char *buf = tok->buffer;
  
  while (off < tok->level && mu_isblank (buf[off]))
    off++;

  if (tok->argc == tok->argmax)
    {
      if (tok->argmax == 0)
	tok->argmax = 16;
      else
	tok->argmax *= 2;
      tok->argp = realloc (tok->argp, tok->argmax * sizeof (tok->argp[0]));
      if (!tok->argp)
	imap4d_bye (ERR_NO_MEM);
    }
  
  if (buf[off] == '"')
    {
      char *start = buf + off + 1;
      char *p = NULL;
      
      while (*start && (p = strchr (start, '"')))
	{
	  if (p == start || p[-1] != '\\')
	    break;
	  start = p + 1;
	}

      if (p)
	{
	  size_t len;
	  off++;
	  len  = unquote (buf + off, p - (buf + off));
	  buf[off + len] = 0;
	  tok->argp[tok->argc++] = off;
	  return p - buf + 1;
	}
    }

  tok->argp[tok->argc++] = off;
  if (ISDELIM (buf[off]))
    return insert_nul (tok, off + 1);

  while (off < tok->level && !mu_isblank (buf[off]))
    {
      if (ISDELIM (buf[off]))
	return insert_nul (tok, off);
      off++;
    }
  buf[off++] = 0;
  
  return off;
}

static void
imap4d_tokbuf_tokenize (struct imap4d_tokbuf *tok, size_t off)
{
  while (off < tok->level)
    off = gettok (tok, off);
}

static void
check_input_err (int rc, size_t sz)
{
  if (rc)
    {
      const char *p;
      if (mu_stream_strerror (istream, &p))
	p = mu_strerror (rc);
      
      mu_diag_output (MU_DIAG_INFO,
		      _("error reading from input file: %s"), p);
      imap4d_bye (ERR_NO_IFILE);
    }
  else if (sz == 0)
    {
      mu_diag_output (MU_DIAG_INFO, _("unexpected eof on input"));
      imap4d_bye (ERR_NO_IFILE);
    }
}

static size_t
imap4d_tokbuf_getline (struct imap4d_tokbuf *tok)
{
  char buffer[512];
  size_t level = tok->level;
  
  do
    {
      size_t len;
      int rc;
      
      rc = mu_stream_sequential_readline (istream,
					  buffer, sizeof (buffer), &len);
      check_input_err (rc, len);
      imap4d_tokbuf_expand (tok, len);
      
      memcpy (tok->buffer + tok->level, buffer, len);
      tok->level += len;
    }
  while (tok->level && tok->buffer[tok->level - 1] != '\n');
  tok->buffer[--tok->level] = 0;
  if (tok->buffer[tok->level - 1] == '\r')
    tok->buffer[--tok->level] = 0;
  return level;
}

void
imap4d_readline (struct imap4d_tokbuf *tok)
{
  int transcript = imap4d_transcript;
  tok->argc = 0;
  tok->level = 0;
  for (;;)
    {
      char *last_arg;
      size_t off = imap4d_tokbuf_getline (tok);
      if (transcript)
        {
          int len;
          char *p = mu_strcasestr (tok->buffer, "LOGIN");
          if (p && p > tok->buffer && mu_isblank (p[-1]))
            {
	      char *q = mu_str_skip_class (p + 5, MU_CTYPE_SPACE);
	      q = mu_str_skip_class_comp (q, MU_CTYPE_SPACE);
              len = q - tok->buffer; 
              mu_diag_output (MU_DIAG_DEBUG,
			      "recv: %*.*s {censored}", len, len,
                              tok->buffer);
             }
           else
             {
               len = strcspn (tok->buffer, "\r\n");
               mu_diag_output (MU_DIAG_DEBUG, "recv: %*.*s", 
                               len, len, tok->buffer);
             }
        }
      imap4d_tokbuf_tokenize (tok, off);
      if (tok->argc == 0)
        break;  
      last_arg = tok->buffer + tok->argp[tok->argc - 1];
      if (last_arg[0] == '{' && last_arg[strlen(last_arg)-1] == '}')
	{
	  int rc;
	  unsigned long number;
	  char *sp = NULL;
	  char *buf;
	  size_t len;
	  
          if (transcript)
            mu_diag_output (MU_DIAG_DEBUG, "(literal follows)");
          transcript = 0;
	  number = strtoul (last_arg + 1, &sp, 10);
	  /* Client can ask for non-synchronised literal,
	     if a '+' is appended to the octet count. */
	  if (*sp == '}')
	    util_send ("+ GO AHEAD\r\n");
	  else if (*sp != '+')
	    break;
	  imap4d_tokbuf_expand (tok, number + 1);
	  off = tok->level;
	  buf = tok->buffer + off;
          len = 0;
          while (len < number)
            {
               size_t sz;
	       rc = mu_stream_sequential_read (istream, 
                                               buf + len, number - len, &sz);
               if (rc || sz == 0)
                 break;
               len += sz;
            }
	  check_input_err (rc, len);
	  len = remove_cr (buf, len);
	  imap4d_tokbuf_unquote (tok, &off, &len);
	  tok->level += len;
	  tok->buffer[tok->level++] = 0;
	  tok->argp[tok->argc - 1] = off;
	}
      else
	break;
    }
}  

struct imap4d_tokbuf *
imap4d_tokbuf_from_string (char *str)
{
  struct imap4d_tokbuf *tok = imap4d_tokbuf_init ();
  tok->buffer = strdup (str);
  if (!tok->buffer)
    imap4d_bye (ERR_NO_MEM);
  tok->level = strlen (str);
  tok->size = tok->level + 1;
  imap4d_tokbuf_tokenize (tok, 0);
  return tok;
}

int
util_trim_nl (char *s, size_t len)
{
  if (s && len > 0 && s[len - 1] == '\n')
    s[--len] = 0;
  if (s && len > 0 && s[len - 1] == '\r')
    s[--len] = 0;
  return len;
}

int
imap4d_getline (char **pbuf, size_t *psize, size_t *pnbytes)
{
  size_t len;
  int rc = mu_stream_sequential_getline (istream, pbuf, psize, &len);
  if (rc == 0)
    {
      char *s = *pbuf;

      if (len == 0)
        {
	  if (imap4d_transcript)
            mu_diag_output (MU_DIAG_DEBUG, "got EOF");
          imap4d_bye (ERR_NO_IFILE);
          /*FIXME rc = ECONNABORTED;*/
        }
      len = util_trim_nl (s, len);
      if (imap4d_transcript)
	mu_diag_output (MU_DIAG_DEBUG, "recv: %s", s);
      if (pnbytes)
	*pnbytes = len;
    }
  return rc;
}
