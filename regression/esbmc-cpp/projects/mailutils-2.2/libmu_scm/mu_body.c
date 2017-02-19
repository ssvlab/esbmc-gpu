/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2010 Free Software
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

#include "mu_scm.h"

static scm_t_bits body_tag;

struct mu_body
{
  mu_body_t body;             /* Message body */
  mu_stream_t stream;         /* Associated stream */
  int offset;              /* Current offset in the stream */
  char *buffer;            /* I/O buffer */
  int bufsize;             /* Size of allocated buffer */
  SCM msg;                 /* Message the body belongs to */		
};

/* Initial buffer size */
#define BUF_SIZE 64

/* SMOB functions: */
static SCM
mu_scm_body_mark (SCM body_smob)
{
  struct mu_body *mbp = (struct mu_body *) SCM_CDR (body_smob);
  return mbp->msg;
}

static scm_sizet
mu_scm_body_free (SCM body_smob)
{
  struct mu_body *mbp = (struct mu_body *) SCM_CDR (body_smob);
  if (mbp->buffer)
    free (mbp->buffer);
  free (mbp);
  return sizeof (struct mu_body);
}

static int
mu_scm_body_print (SCM body_smob, SCM port, scm_print_state * pstate)
{
  struct mu_body *mbp = (struct mu_body *) SCM_CDR (body_smob);
  size_t b_size = 0, b_lines = 0, len = 0;
  char buffer[512];

  mu_body_size (mbp->body, &b_size);
  mu_body_lines (mbp->body, &b_lines);
  buffer[0] = 0;
  mu_body_get_filename (mbp->body, buffer, sizeof (buffer), &len);

  scm_puts ("#<body \"", port);
  scm_puts (buffer, port);
  scm_puts ("\" ", port);

  snprintf (buffer, sizeof (buffer), "%3lu %-5lu",
	    (unsigned long) b_lines, (unsigned long) b_size);
  scm_puts (buffer, port);

  scm_puts (">", port);
  return 1;
}

/* Internal functions: */

int
mu_scm_is_body (SCM scm)
{
  return SCM_NIMP (scm) && (long) SCM_CAR (scm) == body_tag;
}

SCM
mu_scm_body_create (SCM msg, mu_body_t body)
{
  struct mu_body *mbp;

  mbp = scm_gc_malloc (sizeof (struct mu_body), "body");
  mbp->msg = msg;
  mbp->body = body;
  mbp->stream = NULL;
  mbp->offset = 0;
  mbp->buffer = NULL;
  mbp->bufsize = 0;
  SCM_RETURN_NEWSMOB (body_tag, mbp);
}

/* ************************************************************************* */
/* Guile primitives */

SCM_DEFINE_PUBLIC (scm_mu_body_read_line, "mu-body-read-line", 1, 0, 0,
	    (SCM body), 
	    "Read next line from the @var{body}.")
#define FUNC_NAME s_scm_mu_body_read_line
{
  struct mu_body *mbp;
  size_t n, nread;
  int status;
  
  SCM_ASSERT (mu_scm_is_body (body), body, SCM_ARG1, FUNC_NAME);
  mbp = (struct mu_body *) SCM_CDR (body);

  if (!mbp->stream)
    {
      status = mu_body_get_stream (mbp->body, &mbp->stream);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot get body stream",
		      SCM_BOOL_F);
    }

  if (!mbp->buffer)
    {
      mbp->bufsize = BUF_SIZE;
      mbp->buffer = malloc (mbp->bufsize);
      if (!mbp->buffer)
	mu_scm_error (FUNC_NAME, ENOMEM, "Cannot allocate memory", SCM_BOOL_F);
    }

  nread = 0;
  while (1)
    {
      status = mu_stream_readline (mbp->stream, mbp->buffer + nread,
				   mbp->bufsize - nread,
				   mbp->offset, &n);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Error reading from stream", SCM_BOOL_F);
      if (n == 0)
	break;
      nread += n;
      mbp->offset += n;
      if (mbp->buffer[n - 1] != '\n' && n == mbp->bufsize)
	{
	  char *p = realloc (mbp->buffer, mbp->bufsize + BUF_SIZE);
	  if (!p)
	    break;
	  mbp->buffer = p;
	  mbp->bufsize += BUF_SIZE;
	}
      else
	break;
    }

  if (nread == 0)
    return SCM_EOF_VAL;

  return scm_from_locale_string (mbp->buffer);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_body_write, "mu-body-write", 2, 0, 0,
	    (SCM body, SCM text),
	    "Append @var{text} to message @var{body}.")
#define FUNC_NAME s_scm_mu_body_write
{
  char *ptr;
  size_t len, n;
  struct mu_body *mbp;
  int status;
  
  SCM_ASSERT (mu_scm_is_body (body), body, SCM_ARG1, FUNC_NAME);
  mbp = (struct mu_body *) SCM_CDR (body);
  SCM_ASSERT (scm_is_string (text), text, SCM_ARG2, FUNC_NAME);
  
  if (!mbp->stream)
    {
      status = mu_body_get_stream (mbp->body, &mbp->stream);
      if (status)
	mu_scm_error (FUNC_NAME, status,
		      "Cannot get body stream", SCM_BOOL_F);
    }

  ptr = scm_to_locale_string (text);
  len = strlen (ptr);
  status = mu_stream_write (mbp->stream, ptr, len, mbp->offset, &n);
  free (ptr);
  mu_scm_error (FUNC_NAME, status,
		"Error writing to stream", SCM_BOOL_F);
  mbp->offset += n;
  return SCM_BOOL_T;
}
#undef FUNC_NAME

/* Initialize the module */
void
mu_scm_body_init ()
{
  body_tag = scm_make_smob_type ("body", sizeof (struct mu_body));
  scm_set_smob_mark (body_tag, mu_scm_body_mark);
  scm_set_smob_free (body_tag, mu_scm_body_free);
  scm_set_smob_print (body_tag, mu_scm_body_print);

#include "mu_body.x"

}
