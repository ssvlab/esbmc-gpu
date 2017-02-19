/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#include "mail.h"
#include "mailcap.h"

/*
  FIXME:
 decode, this is temporary, until the API on how to present
 mime/attachements etc is less confusing.
 */

struct decode_closure
{
  int select_hdr;
};

static int print_stream (mu_stream_t, FILE *);
static int display_message (mu_message_t, msgset_t *msgset, void *closure);
static int display_submessage (struct mime_descend_closure *closure,
			       void *data);
static int get_content_encoding (mu_header_t hdr, char **value);
static void run_metamail (const char *mailcap, mu_message_t mesg);

int
mail_decode (int argc, char **argv)
{
  msgset_t *msgset;
  struct decode_closure decode_closure;

  if (msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &msgset))
    return 1;

  decode_closure.select_hdr = mu_islower (argv[0][0]);

  util_msgset_iterate (msgset, display_message, (void *)&decode_closure);

  msgset_free (msgset);
  return 0;
}

int
display_message (mu_message_t mesg, msgset_t *msgset, void *arg)
{
  struct decode_closure *closure = arg;
  mu_attribute_t attr = NULL;
  struct mime_descend_closure mclos;
  
  mu_message_get_attribute (mesg, &attr);
  if (mu_attribute_is_deleted (attr))
    return 1;

  mclos.hints = closure->select_hdr ? MDHINT_SELECTED_HEADERS : 0;
  mclos.msgset = msgset;
  mclos.message = mesg;
  mclos.type = NULL;
  mclos.encoding = NULL;
  mclos.parent = NULL;

  mime_descend (&mclos, display_submessage, NULL);

  /* Mark enclosing message as read */
  if (mu_mailbox_get_message (mbox, msgset->msg_part[0], &mesg) == 0)
    util_mark_read (mesg);

  return 0;
}

static void
display_headers (FILE *out, mu_message_t mesg, 
                 const msgset_t *msgset MU_ARG_UNUSED,
		 int select_hdr)
{
  mu_header_t hdr = NULL;

  /* Print the selected headers only.  */
  if (select_hdr)
    {
      size_t num = 0;
      size_t i = 0;
      const char *sptr;

      mu_message_get_header (mesg, &hdr);
      mu_header_get_field_count (hdr, &num);
      for (i = 1; i <= num; i++)
	{
	  if (mu_header_sget_field_name (hdr, i, &sptr))
	    continue;
	  if (mail_header_is_visible (sptr))
	    {
	      fprintf (out, "%s: ", sptr);
	      if (mu_header_sget_field_value (hdr, i, &sptr))
		sptr = "";
	      fprintf (out, "%s\n", sptr);
	    }
	}
      fprintf (out, "\n");
    }
  else /* Print displays all headers.  */
    {
      mu_stream_t stream = NULL;
      if (mu_message_get_header (mesg, &hdr) == 0
	  && mu_header_get_stream (hdr, &stream) == 0)
	print_stream (stream, out);
    }
}

size_t
fprint_msgset (FILE *fp, const msgset_t *msgset)
{
  int i;
  size_t n = 0;
  
  n = fprintf (fp, "%lu", (unsigned long) msgset->msg_part[0]);
  for (i = 1; i < msgset->npart; i++)
    n += fprintf (fp, "[%lu", (unsigned long) msgset->msg_part[i]);
  for (i = 1; i < msgset->npart; i++)
    n += fprintf (fp, "]");
  return n;
}

static void
display_part_header (FILE *out, const msgset_t *msgset,
		     const char *type, const char *encoding)
{
  int size = util_screen_columns () - 3;
  unsigned int i;

  fputc ('+', out);
  for (i = 0; (int)i <= size; i++)
    fputc ('-', out);
  fputc ('+', out);
  fputc ('\n', out);
  fprintf (out, "%s", _("| Message="));
  fprint_msgset (out, msgset);
  fprintf (out, "\n");

  fprintf (out, _("| Type=%s\n"), type);
  fprintf (out, _("| Encoding=%s\n"), encoding);
  fputc ('+', out);
  for (i = 0; (int)i <= size; i++)
    fputc ('-', out);
  fputc ('+', out);
  fputc ('\n', out);
}

int
mime_descend (struct mime_descend_closure *closure,
	      mime_descend_fn fun, void *data)
{
  int status = 0;
  size_t nparts = 0;
  mu_header_t hdr = NULL;
  char *type;
  char *encoding;
  int ismime = 0;
  struct mime_descend_closure subclosure;

  mu_message_get_header (closure->message, &hdr);
  util_get_hdr_value (hdr, MU_HEADER_CONTENT_TYPE, &type);
  if (type == NULL)
    type = xstrdup ("text/plain");
  get_content_encoding (hdr, &encoding);

  closure->type = type;
  closure->encoding = encoding;
  
  subclosure.hints = 0;
  subclosure.parent = closure;
  
  mu_message_is_multipart (closure->message, &ismime);
  if (ismime)
    {
      unsigned int j;

      mu_message_get_num_parts (closure->message, &nparts);

      for (j = 1; j <= nparts; j++)
	{
	  mu_message_t message = NULL;

	  if (mu_message_get_part (closure->message, j, &message) == 0)
	    {
	      msgset_t *set = msgset_expand (msgset_dup (closure->msgset),
					     msgset_make_1 (j));
	      subclosure.msgset = set;
	      subclosure.message = message;
	      status = mime_descend (&subclosure, fun, data);
	      msgset_free (set);
	      if (status)
		break;
	    }
	}
    }
  else if (mu_c_strncasecmp (type, "message/rfc822", strlen (type)) == 0)
    {
      mu_message_t submsg = NULL;

      if (mu_message_unencapsulate (closure->message, &submsg, NULL) == 0)
	{
	  subclosure.hints = MDHINT_SELECTED_HEADERS;
	  subclosure.msgset = closure->msgset;
	  subclosure.message = submsg;
	  status = mime_descend (&subclosure, fun, data);
	}
    }
  else
    status = fun (closure, data);

  closure->type = NULL;
  closure->encoding = NULL;
  
  free (type);
  free (encoding);

  return status;
}

static int
display_submessage (struct mime_descend_closure *closure, void *data)
{
  char *tmp;
  
  if (mailvar_get (&tmp, "metamail", mailvar_type_string, 0) == 0)
    {
      /* If `metamail' is set to a string, treat it as command line
	 of external metamail program. */
      run_metamail (tmp, closure->message);
    }
  else
    {
      int builtin_display = 1;
      mu_body_t body = NULL;
      mu_stream_t b_stream = NULL;
      mu_stream_t d_stream = NULL;
      mu_stream_t stream = NULL;
      mu_header_t hdr = NULL;
      
      mu_message_get_body (closure->message, &body);
      mu_message_get_header (closure->message, &hdr);
      mu_body_get_stream (body, &b_stream);

      /* Can we decode.  */
      if (mu_filter_create(&d_stream, b_stream, closure->encoding,
			   MU_FILTER_DECODE, 
			   MU_STREAM_READ|MU_STREAM_NO_CLOSE) == 0)
	stream = d_stream;
      else
	stream = b_stream;
      
      display_part_header (ofile, closure->msgset,
			   closure->type, closure->encoding);
      
      /* If `metamail' is set to true, enable internal mailcap
	 support */
      if (mailvar_get (NULL, "metamail", mailvar_type_boolean, 0) == 0)
	{
	  char *no_ask = NULL;
	  int debug = 0;
	  
	  mailvar_get (&no_ask, "mimenoask", mailvar_type_string, 0);
	  if (mailvar_get (&debug, "verbose", mailvar_type_boolean, 0) == 0
	      && debug)
	    debug = 9;
	  
	  builtin_display = display_stream_mailcap (NULL, stream, hdr, no_ask,
						    interactive, 0, debug);
	}
      
      if (builtin_display)
	{
	  size_t lines = 0;
	  int pagelines = util_get_crt ();
	  FILE *out;
	  
	  mu_message_lines (closure->message, &lines);
	  if (pagelines && lines > pagelines)
	    out = popen (getenv ("PAGER"), "w");
	  else
	    out = ofile;
	  
	  display_headers (out, closure->message, closure->msgset,
			   closure->hints & MDHINT_SELECTED_HEADERS);
	  
	  print_stream (stream, out);
	  
	  if (out != ofile)
	    pclose (out);
	}
      if (d_stream)
	mu_stream_destroy (&d_stream, NULL);
    }
  
  return 0;
}

static int
print_stream (mu_stream_t stream, FILE *out)
{
  char buffer[512];
  off_t off = 0;
  size_t n = 0;

  while (mu_stream_read (stream, buffer, sizeof (buffer) - 1, off, &n) == 0
	 && n != 0)
    {
      if (ml_got_interrupt())
	{
	  util_error(_("\nInterrupt"));
	  break;
	}
      buffer[n] = '\0';
      fprintf (out, "%s", buffer);
      off += n;
    }
  return 0;
}

static int
get_content_encoding (mu_header_t hdr, char **value)
{
  char *encoding = NULL;
  util_get_hdr_value (hdr, MU_HEADER_CONTENT_TRANSFER_ENCODING, &encoding);
  if (encoding == NULL || *encoding == '\0')
    {
      if (encoding)
	free (encoding);
      encoding = strdup ("7bit"); /* Default.  */
    }
  *value = encoding;
  return 0;
}

/* Run `metamail' program MAILCAP_CMD on the message MESG */
static void
run_metamail (const char *mailcap_cmd, mu_message_t mesg)
{
  pid_t pid;
  struct sigaction ignore;
  struct sigaction saveintr;
  struct sigaction savequit;
  sigset_t chldmask;
  sigset_t savemask;
  
  ignore.sa_handler = SIG_IGN;	/* ignore SIGINT and SIGQUIT */
  ignore.sa_flags = 0;
  sigemptyset (&ignore.sa_mask);
  if (sigaction (SIGINT, &ignore, &saveintr) < 0)
    {
      util_error ("sigaction: %s", strerror (errno));
      return;
    }      
  if (sigaction (SIGQUIT, &ignore, &savequit) < 0)
    {
      util_error ("sigaction: %s", strerror (errno));
      sigaction (SIGINT, &saveintr, NULL);
      return;
    }      
      
  sigemptyset (&chldmask);	/* now block SIGCHLD */
  sigaddset (&chldmask, SIGCHLD);

  if (sigprocmask (SIG_BLOCK, &chldmask, &savemask) < 0)
    {
      sigaction (SIGINT, &saveintr, NULL);
      sigaction (SIGQUIT, &savequit, NULL);
      return;
    }
  
  pid = fork ();
  if (pid < 0)
    {
      util_error ("fork: %s", strerror (errno));
    }
  else if (pid == 0)
    {
      /* Child process */
      int status;
      mu_stream_t stream;

      do /* Fake loop to avoid gotos */
	{
	  mu_stream_t pstr;
	  char buffer[512];
	  size_t n;
	  char *no_ask;
	  
	  setenv ("METAMAIL_PAGER", getenv ("PAGER"), 0);
	  if (mailvar_get (&no_ask, "mimenoask", mailvar_type_string, 0))
	    setenv ("MM_NOASK", no_ask, 1);
	  
	  status = mu_message_get_stream (mesg, &stream);
	  if (status)
	    {
	      mu_error ("mu_message_get_stream: %s", mu_strerror (status));
	      break;
	    }

	  status = mu_prog_stream_create (&pstr, mailcap_cmd, MU_STREAM_WRITE);
	  if (status)
	    {
	      mu_error ("mu_prog_stream_create: %s", mu_strerror (status));
	      break;
	    }

	  status = mu_stream_open (pstr);
	  if (status)
	    {
	      mu_error ("mu_stream_open: %s", mu_strerror (status));
	      break;
	    }
	  
	  while (mu_stream_sequential_read (stream,
					 buffer, sizeof buffer - 1,
					 &n) == 0
		 && n > 0)
	    mu_stream_sequential_write (pstr, buffer, n);

	  mu_stream_close (pstr);
	  mu_stream_destroy (&pstr, mu_stream_get_owner (pstr));
	  exit (0);
	}
      while (0);
      
      abort ();
    }
  else
    {
      int status;
      /* Master process */
      
      while (waitpid (pid, &status, 0) < 0)
	if (errno != EINTR)
	  break;
    }

  /* Restore the signal handlers */
  sigaction (SIGINT, &saveintr, NULL);
  sigaction (SIGQUIT, &savequit, NULL);
  sigprocmask (SIG_SETMASK, &savemask, NULL);
}
