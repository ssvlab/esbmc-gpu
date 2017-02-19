/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
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

#include <frm.h>

char *show_field;   /* Show this header field instead of the default
			      `From: Subject:' pair. -f option */
int show_to;        /* Additionally display To: field. -l option */ 
int show_number;    /* Prefix each line with the message number. -n */
int frm_debug;



/* Get the number of columns on the screen
   First try an ioctl() call, not all shells set the COLUMNS environ.
   If ioctl does not succeed on stdout, try it on /dev/tty, as we
   may work via a pipe.
   
   This function was taken from mail/util.c. It should probably reside
   in the library */
int
util_getcols (void)
{
  struct winsize ws;
  
  ws.ws_col = ws.ws_row = 0;
  if (ioctl (1, TIOCGWINSZ, (char *) &ws) < 0)
    {
      int fd = open ("/dev/tty", O_RDWR);
      ioctl (fd, TIOCGWINSZ, (char *) &ws);
      close (fd);
    }
  if (ws.ws_row == 0)
    {
      const char *columns = getenv ("COLUMNS");
      if (columns)
	ws.ws_col = strtol (columns, NULL, 10);
    }
  return ws.ws_col;
}


/* Charset magic */
static char *output_charset = NULL;

const char *
get_charset ()
{
  if (!output_charset)
    {
      char *tmp;
      const char *str = NULL;
      char locale[32];
      
      memset (locale, 0, sizeof (locale));

      /* Try to deduce the charset from LC_ALL or LANG variables */

      tmp = getenv ("LC_ALL");
      if (!tmp)
	tmp = getenv ("LANG");

      if (tmp)
	{
	  char *sp = NULL;
	  char *lang;
	  char *terr;

	  strncpy (locale, tmp, sizeof (locale) - 1);
	  
	  lang = strtok_r (locale, "_", &sp);
	  terr = strtok_r (NULL, ".", &sp);
	  str = strtok_r (NULL, "@", &sp);

	  if (!str)
	    str = mu_charset_lookup (lang, terr);
	}
      
      if (!str)
	str = "ASCII";

      output_charset = xstrdup (str);
    }
  return output_charset;
}


/* BIDI support (will be moved to lib when it's ready) */
#ifdef HAVE_LIBFRIBIDI

# ifdef HAVE_FRIBIDI_WCWIDTH
#  define mu_fribidi_wcwidth fribidi_wcwidth
# else
#  if defined(HAVE_WCHAR_H) && defined(HAVE_WCWIDTH)
#   include <wchar.h>
#   define mu_fribidi_wcwidth(c) wcwidth((wchar_t)c)
#  else
#   undef HAVE_LIBFRIBIDI
#  endif
# endif
#endif

#ifdef HAVE_LIBFRIBIDI

static int fb_charset_num = -1;
FriBidiChar *logical;
char *outstring;
size_t logical_size;

void
alloc_logical (size_t size)
{
  logical = xmalloc (size * sizeof (logical[0]));
  logical_size = size;
  outstring = xmalloc (size);
}

void
puts_bidi (char *string)
{
  if (fb_charset_num == -1)
    {
      fb_charset_num = fribidi_parse_charset ((char*) get_charset ());
      if (fb_charset_num && frm_debug)
	mu_error (_("fribidi failed to recognize charset `%s'"),
		  get_charset ());
    }
  
  if (fb_charset_num == 0)
    puts (string);
  else
    {
      FriBidiStrIndex len;
      FriBidiCharType base = FRIBIDI_TYPE_ON;
      fribidi_boolean log2vis;
      
      static FriBidiChar *visual;
      static size_t visual_size;
      
      
      len = fribidi_charset_to_unicode (fb_charset_num,
					string, strlen (string),
					logical);

      if (len + 1 > visual_size)
	{
	  visual_size = len + 1;
	  visual = xrealloc (visual, visual_size * sizeof *visual);
	}
      
      /* Create a bidi string. */
      log2vis = fribidi_log2vis (logical, len, &base,
				 /* output */
				 visual, NULL, NULL, NULL);

      if (log2vis)
	{
	  FriBidiStrIndex idx, st;
	  FriBidiStrIndex new_len;
	  
	  for (idx = 0; idx < len;)
	    {
	      FriBidiStrIndex wid, inlen;
	      
	      wid = 3 * logical_size;
	      st = idx;

	      if (fb_charset_num != FRIBIDI_CHARSET_CAP_RTL)
		{
		  while (wid > 0 && idx < len)
		    wid -= mu_fribidi_wcwidth (visual[idx++]);
		}
	      else
		{
		  while (wid > 0 && idx < len)
		    {
		      wid--;
		      idx++;
		    }
		}
	      
	      if (wid < 0 && idx > st + 1)
		idx--;
	      inlen = idx - st;

	      new_len = fribidi_unicode_to_charset (fb_charset_num,
						    visual + st, inlen,
						    outstring);
	      printf ("%s", outstring);
	    }
	  putchar ('\n');
	}
      else
	{
	  /* Print the string as is */
	  puts (string);
	}
    }
}
#else
# define alloc_logical(s)
# define puts_bidi puts
#endif


/* Output functions */

/* Number of columns in output:

     Maximum     4     message number, to, from, subject   -ln
     Default     2     from, subject                       [none]
     Minimum     1     FIELD                               -f FIELD
*/

static int numfields;      /* Number of output fields */
static int fieldwidth[4];  /* Field start positions */
static char *linebuf;      /* Output line buffer */
static size_t linemax;     /* Size of linebuf */
static size_t linepos;     /* Position in the output line buffer */
static int curfield;       /* Current output field */
static int nextstart;      /* Start position of the next field */
static int curcol;         /* Current output column */

typedef void (*fmt_formatter) (const char *fmt, ...);

static fmt_formatter format_field;

void
print_line ()
{
  if (linebuf)
    {
      puts_bidi (linebuf);
      linebuf[0] = 0;
      linepos = 0;
      curcol = nextstart = 0;
    }
  else
    putchar ('\n');
  curfield = 0;
}

void
format_field_simple (const char *fmt, ...)
{
  va_list ap;
  if (curfield++)
    putchar ('\t');
  va_start (ap, fmt);
  vprintf (fmt, ap);
  va_end (ap);
}

void
format_field_align (const char *fmt, ...)
{
  size_t n, width;
  va_list ap;

  va_start (ap, fmt);
  if (nextstart != 0)
    {
      if (curcol >= nextstart)
	{
	  if (curfield == numfields - 1)
	    {
	      puts_bidi (linebuf);
	      linepos = 0;
	      printf ("%*s", nextstart, "");
	    }
	  else
	    {
	      linebuf[linepos++] = ' ';
	      curcol++;
	    }
	}
      else if (nextstart != curcol)
	{
	  /* align to field start */
	  n = snprintf (linebuf + linepos, linemax - linepos,
			"%*s", nextstart - curcol, "");
	  linepos += n;
	  curcol = nextstart;
	}
    }

  n = vsnprintf (linebuf + linepos, linemax - linepos, fmt, ap);
  va_end (ap);

  /* Compute output width */
  if (curfield == numfields - 1)
    {
      for ( ; n > 0; n--)
	{
	  int c = linebuf[linepos + n];
	  linebuf[linepos + n] = 0;
	  width = mbswidth (linebuf + linepos, 0);
	  if (width <= fieldwidth[curfield])
	    break;
	  linebuf[linepos + n] = c;
	}
    }
  else
    width = mbswidth (linebuf + linepos, 0);

  /* Increment counters */
  linepos += n;
  curcol += width;
  nextstart += fieldwidth[curfield++];
}

void
init_output (size_t s)
{
  int i;
  size_t width = 0;

  if (s == 0)
    {
      format_field = format_field_simple;
      return;
    }
  
  format_field = format_field_align;
	  
  /* Allocate the line buffer */
  linemax = s * MB_LEN_MAX + 1;
  linebuf = xmalloc (linemax);
  alloc_logical (s);
	  
  /* Set up column widths */
  if (show_number)
    fieldwidth[numfields++] = 5;
  
  if (show_to)
    fieldwidth[numfields++] = 20;
  
  if (show_field)
    fieldwidth[numfields++] = 0;
  else
    {
      fieldwidth[numfields++] = 20;
      fieldwidth[numfields++] = 0;
    }
  
  for (i = 0; i < numfields; i++)
    width += fieldwidth[i];
  
  fieldwidth[numfields-1] = util_getcols () - width;
}


/*
  FIXME: Generalize this function and move it
  to `mailbox/locale.c'. Do the same with the one
  from `from/from.c' and `mail/util.c'...
*/
static char *
rfc2047_decode_wrapper (const char *buf, size_t buflen)
{
  int rc;
  char *tmp;
  const char *charset = get_charset ();
  
  if (strcmp (charset, "ASCII") == 0)
    return strdup (buf);

  rc = mu_rfc2047_decode (charset, buf, &tmp);
  if (rc)
    {
      if (frm_debug)
	mu_error (_("cannot decode line `%s': %s"),
		  buf, mu_strerror (rc));
      return strdup (buf);
    }

  return tmp;
}

/* Retrieve the Personal Name from the header To: or From:  */
static int
get_personal (mu_header_t hdr, const char *field, char **personal)
{
  char *hfield;
  int status;

  status = mu_header_aget_value_unfold (hdr, field, &hfield);
  if (status == 0)
    {
      mu_address_t address = NULL;
      char *s = NULL;
      
      mu_address_create (&address, hfield);
      
      mu_address_aget_personal (address, 1, &s);
      mu_address_destroy (&address);
      if (s == NULL)
	s = hfield;
      else
	free (hfield);

      *personal = rfc2047_decode_wrapper (s, strlen (s));
      free (s);
    }
  return status;
}

/* Observer action used to perform mailbox scanning. See the comment
   to frm_scan below.

   FIXME: The observer action paradygm does not allow for making
   procedure-data closure, as it should. So, for the time being
   the following static variables are used instead: */

static frm_select_t select_message;  /* Message selection function */
static size_t msg_index;             /* Index (1-based) of the current
					message */

/* Observable action is being called on discovery of each message. */
/* FIXME: The format of the display is poorly done, please correct.  */
static int
action (mu_observer_t o, size_t type, void *data, void *action_data)
{
  int status;

  switch (type)
    {
    case MU_EVT_MESSAGE_ADD:
      {
	mu_mailbox_t mbox = mu_observer_get_owner (o);
	mu_message_t msg = NULL;
	mu_header_t hdr = NULL;
	mu_attribute_t attr = NULL;

	msg_index++;
	
	mu_mailbox_get_message (mbox, msg_index, &msg);
	mu_message_get_attribute (msg, &attr);
	mu_message_get_header (msg, &hdr);

	if (!select_message (msg_index, msg))
	  break;

	if (show_number)
	  format_field ("%4lu:", (u_long) msg_index);

	if (show_to)
	  {
	    char *hto;
	    status = get_personal (hdr, MU_HEADER_TO, &hto);

	    if (status == 0)
	      {
		format_field ("(%s)", hto);
		free (hto);
	      }
	    else
	      format_field ("(none)");
	  }

	if (show_field) /* FIXME: This should be also mu_rfc2047_decode. */
	  {
	    char *hfield;
	    status = mu_header_aget_value_unfold (hdr, show_field, &hfield);
	    if (status == 0)
	      {
		format_field ("%s", hfield);
		free (hfield);
	      }
	    else
	      format_field ("");
	  }
	else
	  {
	    char *tmp;
	    status = get_personal (hdr, MU_HEADER_FROM, &tmp);
	    if (status == 0)
	      {
		format_field ("%s", tmp);
		free (tmp);
	      }
	    else
	      format_field ("");

	    status = mu_header_aget_value_unfold (hdr, MU_HEADER_SUBJECT,
					       &tmp);
	    if (status == 0)
	      {
		char *s = rfc2047_decode_wrapper (tmp, strlen (tmp));
		format_field ("%s", s);
		free (s);
		free (tmp);
	      }
	  }
	print_line ();
	break;
      }

    case MU_EVT_MAILBOX_PROGRESS:
      /* Noop.  */
      break;
    }
  return 0;
}

static void
frm_abort (mu_mailbox_t *mbox)
{
  int status;
  
  if ((status = mu_mailbox_close (*mbox)) != 0)
    {
      mu_url_t url;
      mu_mailbox_get_url (*mbox, &url);
      mu_error (_("could not close mailbox `%s': %s"),
		mu_url_to_string (url), mu_strerror (status));
      exit (3);
    }
  
  mu_mailbox_destroy (mbox);
  exit (3);
}

/* Scan the mailbox MAILBOX_NAME using FUN as the selection function.
   FUN takes as its argument message number and the message itself
   (mu_message_t). It returns non-zero if that message is to be displayed
   and zero otherwise.

   Upon finishing scanning, the function places the overall number of
   the messages processed into the memory location pointed to by
   TOTAL */
void
frm_scan (char *mailbox_name, frm_select_t fun, size_t *total)
{
  mu_mailbox_t mbox;
  int status;
  mu_url_t url;
  
  status = mu_mailbox_create_default (&mbox, mailbox_name);
  if (status != 0)
    {
      if (mailbox_name)
	mu_error (_("could not create mailbox `%s': %s"),
		  mailbox_name,  mu_strerror (status));
      else
	mu_error (_("could not create default mailbox: %s"),
		  mu_strerror (status));
      exit (3);
    }

  if (frm_debug)
    {
      mu_debug_t debug;
      mu_mailbox_get_debug (mbox, &debug);
      mu_debug_set_level (debug, MU_DEBUG_LEVEL_UPTO (MU_DEBUG_PROT));
    }

  mu_mailbox_get_url (mbox, &url);

  status = mu_mailbox_open (mbox, MU_STREAM_READ);
  if (status == ENOENT)
    *total = 0;
  else if (status != 0)
    {
      mu_error (_("could not open mailbox `%s': %s"),
		mu_url_to_string (url), mu_strerror (status));
      mu_mailbox_destroy (&mbox);
      exit (3);
    }
  else
    {
      mu_observer_t observer;
      mu_observable_t observable;

      select_message = fun;
      msg_index = 0;
      
      mu_observer_create (&observer, mbox);
      mu_observer_set_action (observer, action, mbox);
      mu_mailbox_get_observable (mbox, &observable);
      mu_observable_attach (observable, MU_EVT_MESSAGE_ADD, observer);
      
      status = mu_mailbox_scan (mbox, 1, total);
      if (status != 0)
	{
	  mu_error (_("could not scan mailbox `%s': %s."),
		    mu_url_to_string (url), mu_strerror (status));
	  frm_abort (&mbox);
	}
      
      mu_observable_detach (observable, observer);
      mu_observer_destroy (&observer, mbox);
      
      if ((status = mu_mailbox_close (mbox)) != 0)
	{
	  mu_error (_("could not close mailbox `%s': %s"),
		    mu_url_to_string (url), mu_strerror (status));
	  exit (3);
	}
    }
  mu_mailbox_destroy (&mbox);
}
