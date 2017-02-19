/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2006, 2007, 2010 Free Software
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

static scm_t_bits mime_tag;

struct mu_mime
{
  mu_mime_t mime;
  SCM owner;
};

/* SMOB functions: */

static SCM
mu_scm_mime_mark (SCM mime_smob)
{
  struct mu_mime *mum = (struct mu_mime *) SCM_CDR (mime_smob);
  return mum->owner;
}

static scm_sizet
mu_scm_mime_free (SCM mime_smob)
{
  struct mu_mime *mum = (struct mu_mime *) SCM_CDR (mime_smob);
  mu_mime_destroy (&mum->mime);
  free (mum);
  return sizeof (struct mu_mime);
}

static int
mu_scm_mime_print (SCM mime_smob, SCM port, scm_print_state * pstate)
{
  struct mu_mime *mum = (struct mu_mime *) SCM_CDR (mime_smob);
  size_t nparts = 0;
  
  mu_mime_get_num_parts (mum->mime, &nparts);
  
  scm_puts ("#<mime ", port);
  scm_intprint (nparts, 10, port);
  scm_putc ('>', port);
  
  return 1;
}

/* Internal calls: */

SCM
mu_scm_mime_create (SCM owner, mu_mime_t mime)
{
  struct mu_mime *mum;

  mum = scm_gc_malloc (sizeof (struct mu_mime), "mime");
  mum->owner = owner;
  mum->mime = mime;
  SCM_RETURN_NEWSMOB (mime_tag, mum);
}

mu_mime_t
mu_scm_mime_get (SCM MIME)
{
  struct mu_mime *mum = (struct mu_mime *) SCM_CDR (MIME);
  return mum->mime;
}

int
mu_scm_is_mime (SCM scm)
{
  return SCM_NIMP (scm) && (long) SCM_CAR (scm) == mime_tag;
}

/* ************************************************************************* */
/* Guile primitives */

SCM_DEFINE_PUBLIC (scm_mu_mime_create, "mu-mime-create", 0, 2, 0,
		   (SCM flags, SCM mesg),
"Creates a new @acronym{MIME} object.  Both arguments are optional.\n"
"@var{Flags} specifies the type of the object to create (@samp{0} is a\n"
"reasonable value).  @var{mesg} gives the message to create the\n"
"@acronym{MIME} object from.")
#define FUNC_NAME s_scm_mu_mime_create
{
  mu_message_t msg = NULL;
  mu_mime_t mime;
  int fl;
  int status;
  
  if (scm_is_bool (flags))
    {
      /*if (flags == SCM_BOOL_F)*/
      fl = 0;
    }
  else
    {
      SCM_ASSERT (scm_is_integer (flags), flags, SCM_ARG1, FUNC_NAME);
      fl = scm_to_int (flags);
    }
  
  if (!SCM_UNBNDP (mesg))
    {
      SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG2, FUNC_NAME);
      msg = mu_scm_message_get (mesg);
    }
  
  status = mu_mime_create (&mime, msg, fl);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot create MIME object", SCM_BOOL_F);
  
  return mu_scm_mime_create (mesg, mime);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_mime_multipart_p, "mu-mime-multipart?", 1, 0, 0,
		   (SCM mime),
"Returns @code{#t} if @var{mime} is a multipart object.\n")
#define FUNC_NAME s_scm_mu_mime_multipart_p
{
  SCM_ASSERT (mu_scm_is_mime (mime), mime, SCM_ARG1, FUNC_NAME);
  return mu_mime_is_multipart (mu_scm_mime_get (mime)) ? SCM_BOOL_T : SCM_BOOL_F;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_mime_get_num_parts, "mu-mime-get-num-parts", 1, 0, 0,
		   (SCM mime),
"Returns number of parts in the @acronym{MIME} object @var{mime}.")
#define FUNC_NAME s_scm_mu_mime_get_num_parts
{
  mu_mime_t mimeobj;
  size_t nparts;
  int status;
  
  SCM_ASSERT (mu_scm_is_mime (mime), mime, SCM_ARG1, FUNC_NAME);
  mimeobj = mu_scm_mime_get (mime);
  status = mu_mime_get_num_parts (mimeobj, &nparts);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot count MIME parts", SCM_BOOL_F);
  return scm_from_size_t (nparts);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_mime_get_part, "mu-mime-get-part", 2, 0, 0,
		   (SCM mime, SCM num),
"Returns @var{num}th part from the @acronym{MIME} object @var{mime}.")
#define FUNC_NAME s_scm_mu_mime_get_part
{
  mu_message_t msg = NULL;
  int status;
  
  SCM_ASSERT (mu_scm_is_mime (mime), mime, SCM_ARG1, FUNC_NAME);
  SCM_ASSERT (scm_is_integer (num), num, SCM_ARG2, FUNC_NAME);
  
  status = mu_mime_get_part (mu_scm_mime_get (mime),
			     scm_to_int (num), &msg);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get part ~A from MIME object ~A",
		  scm_list_2 (num, mime));
  
  return mu_scm_message_create (mime, msg);
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_mime_add_part, "mu-mime-add-part", 2, 0, 0,
		   (SCM mime, SCM mesg),
"Adds message @var{mesg} to the @acronym{MIME} object @var{mime}.")
#define FUNC_NAME s_scm_mu_mime_add_part
{
  mu_mime_t mimeobj;
  mu_message_t msg;
  int status;
  
  SCM_ASSERT (mu_scm_is_mime (mime), mime, SCM_ARG1, FUNC_NAME);
  SCM_ASSERT (mu_scm_is_message (mesg), mesg, SCM_ARG2, FUNC_NAME);
  mimeobj = mu_scm_mime_get (mime);
  msg = mu_scm_message_get (mesg);

  status = mu_mime_add_part (mimeobj, msg);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot add new part to MIME object ~A",
		  scm_list_1 (mime));
  
  mu_scm_message_add_owner (mesg, mime);
  
  return SCM_BOOL_T;
}
#undef FUNC_NAME

SCM_DEFINE_PUBLIC (scm_mu_mime_get_message, "mu-mime-get-message", 1, 0, 0,
		   (SCM mime),
"Converts @acronym{MIME} object @var{mime} to a message.\n")
#define FUNC_NAME s_scm_mu_mime_get_message
{
  mu_mime_t mimeobj;
  mu_message_t msg;
  int status;
  
  SCM_ASSERT (mu_scm_is_mime (mime), mime, SCM_ARG1, FUNC_NAME);
  mimeobj = mu_scm_mime_get (mime);
  status = mu_mime_get_message (mimeobj, &msg);
  if (status)
    mu_scm_error (FUNC_NAME, status,
		  "Cannot get message from MIME object ~A",
		  scm_list_1 (mime));

  return mu_scm_message_create (mime, msg);
}
#undef FUNC_NAME

  
/* Initialize the module */

void
mu_scm_mime_init ()
{
  mime_tag = scm_make_smob_type ("mime", sizeof (struct mu_mime));
  scm_set_smob_mark (mime_tag, mu_scm_mime_mark);
  scm_set_smob_free (mime_tag, mu_scm_mime_free);
  scm_set_smob_print (mime_tag, mu_scm_mime_print);

#include "mu_mime.x"

}

