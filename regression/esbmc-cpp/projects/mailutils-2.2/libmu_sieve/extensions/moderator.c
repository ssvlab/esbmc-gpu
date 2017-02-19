/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2006, 2007, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with GNU Mailutils; if not, write to the Free
   Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

/* Moderator robot for Mailman-driven mail archives.
   Mailman moderation request is a MIME message consisting of the three parts:

   1  text/plain      Introduction for the human reader.
   2  message/rfc822  Original submission.
   3  message/rfc822  Mailman control message.

   Replying to part 3 (keeping the subject intact) instructs Mailman to
   discard the original submission.
   Replying to part 3 while adding an `Approved:' header with the list
   password in it approves the submission.

   Syntax:

     moderator [:keep]
               [:address <address: string>]
	       [:source <sieve-file: string>]

   The moderator action spawns an inferior Sieve machine and filters the
   original submission (part 2) through it. If the inferior machine marks
   the message as deleted, the action replies to the control message,
   thereby causing the submission to be discarded. The From: address of the
   reply can be modified using :address tag. After discarding the message,
   moderator marks it as deleted, unless it is given :keep tag.

   If :source tag is given, its argument sieve-file specifies the Sieve
   source file to be used on the message. Otherwise, moderator will create
   a copy of the existing Sieve machine.

   The action checks the message structure: it will bail out if the message
   does not have exactly 3 MIME parts, or if parts 2 and 3 are not of
   message/rfc822. It is the responsibility of the caller to make sure
   the message is actually a valid Mailman moderation request, for example:

   if allof(header :is "Sender" "mailman-bounces@gnu.org",
            header :is "X-List-Administrivia" "yes")
     {
        moderator :source "~/.sieve/mailman.sv";
     }
   
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif  

#include <unistd.h>
#include <mailutils/sieve.h>
#include <stdlib.h>

static int
moderator_filter_message (mu_sieve_machine_t mach, mu_list_t tags,
			  mu_message_t msg, int *pdiscard)
{
  int rc;
  mu_sieve_machine_t newmach;
  mu_attribute_t attr;
  mu_sieve_value_t *arg;
  
  if (mu_sieve_tag_lookup (tags, "source", &arg))
    {
      rc = mu_sieve_machine_inherit (mach, &newmach);
      if (rc)
	{
	  mu_sieve_error (mach, _("cannot initialize sieve machine: %s"),
			  mu_strerror (rc));
	  return 1;
	}
      /* FIXME: This should be configurable:
	   moderator :inherit
	   moderator :debug 2
	   ...
      */
      
      rc = mu_sieve_compile (newmach, arg->v.string);
      if (rc)
	mu_sieve_error (mach, _("cannot compile source `%s'"), arg->v.string);
    }
  else
    rc = mu_sieve_machine_dup (mach, &newmach);

  if (rc)
    return rc;

  mu_message_get_attribute (msg, &attr);
  mu_attribute_unset_deleted (attr);
  
  rc = mu_sieve_message (newmach, msg);

  if (rc)
    mu_sieve_error (newmach, _("failed to run inferior sieve machine"));
  else
    *pdiscard = mu_attribute_is_deleted (attr);
  
  mu_sieve_machine_destroy (&newmach);

  return rc;
}

static int
copy_header (mu_sieve_machine_t mach,
	     mu_header_t to_hdr, char *to, mu_header_t from_hdr, char *from)
{
  int rc;
  char *value = NULL;
  if ((rc = mu_header_aget_value (from_hdr, from, &value)))
    {
      mu_sieve_error (mach, _("cannot get `%s:' header: %s"),
		      from, mu_strerror (rc));
      return rc;
    }
  rc = mu_header_set_value (to_hdr, to, value, 0);
  free (value);
  return rc;
}


static int
moderator_discard_message (mu_sieve_machine_t mach, mu_message_t request,
			   const char *from)
{
  int rc;
  mu_message_t reply;
  mu_header_t repl_hdr, req_hdr;
  mu_mailer_t mailer;
  
  rc = mu_message_create (&reply, NULL);
  if (rc)
    return rc;
  rc = mu_message_get_header (reply, &repl_hdr);
  if (rc)
    {
      mu_message_destroy (&reply, NULL);
      return rc;
    }
  
  rc = mu_message_get_header (request, &req_hdr);
  if (rc)
    {
      mu_message_destroy (&reply, NULL);
      return rc;
    }

  if (copy_header (mach, repl_hdr, MU_HEADER_TO, req_hdr, MU_HEADER_FROM)
      || copy_header (mach,
		      repl_hdr, MU_HEADER_SUBJECT, req_hdr, MU_HEADER_SUBJECT))
    {
      mu_message_destroy (&reply, NULL);
      return rc;
    }

  if (from)
    mu_header_set_value (repl_hdr, MU_HEADER_FROM, from, 0);

  mailer = mu_sieve_get_mailer (mach);
  rc = mu_mailer_open (mailer, 0);
  if (rc)
    mu_sieve_error (mach, _("cannot open mailer: %s"),
		    mu_strerror (rc));
  else
    {
      rc = mu_mailer_send_message (mailer, reply, NULL, NULL);
      mu_mailer_close (mailer);

      if (rc)
	mu_sieve_error (mach, _("cannot send message: %s"),
			mu_strerror (rc));
    }
  mu_message_destroy (&reply, NULL);
  return rc;
}

int
moderator_message_get_part (mu_sieve_machine_t mach,
			    mu_message_t msg, size_t index, mu_message_t *pmsg)
{
  int rc;
  mu_message_t tmp;
  mu_header_t hdr = NULL;
  char *value;

  if ((rc = mu_message_get_part (msg, index, &tmp)))
    {
      mu_sieve_error (mach, _("cannot get message part #%lu: %s"),
		      (unsigned long) index, mu_strerror (rc));
      return 1;
    }
  
  mu_message_get_header (tmp, &hdr);
  if (mu_header_aget_value (hdr, MU_HEADER_CONTENT_TYPE, &value) == 0
      && memcmp (value, "message/rfc822", 14) == 0)
    {
      mu_stream_t str;
      mu_body_t body;

      free (value);
      mu_message_get_body (tmp, &body);
      mu_body_get_stream (body, &str);

      rc = mu_stream_to_message (str, pmsg);
      if (rc)
	{
	  mu_sieve_error (mach,
			  _("cannot convert MIME part stream to message: %s"),
			  mu_strerror (rc));
	  return 1;
	}
    }
  else if (value)
    {
      mu_sieve_error (mach,
		      _("expected message type message/rfc822, but found %s"),
		      value);
      free (value);
      return 1;
    }
  else
    {
      mu_sieve_error (mach, _("no Content-Type header found"));
      return 1;
    }
  return 0;
}

static int
moderator_action (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  mu_message_t msg, orig;
  int rc;
  size_t nparts = 0;
  int discard = 0;
  int ismime;
  
  if (mu_sieve_get_debug_level (mach) & MU_SIEVE_DEBUG_TRACE)
    {
      mu_sieve_locus_t locus;
      mu_sieve_get_locus (mach, &locus);
      mu_sieve_debug (mach, "%s:%lu: moderator_test %lu\n",
		      locus.source_file,
		      (u_long) locus.source_line,
		      (u_long) mu_sieve_get_message_num (mach));
    }

  msg = mu_sieve_get_message (mach);
  mu_message_is_multipart (msg, &ismime);

  if (!ismime)
    {
      mu_sieve_error (mach, _("message is not multipart"));
      mu_sieve_abort (mach);
    }

  mu_message_get_num_parts (msg, &nparts);

  if (nparts != 3) /* Mailman moderation requests have three parts */
    {
      mu_sieve_error (mach, _("expected 3 parts, but found %lu"),
		      (unsigned long) nparts);
      mu_sieve_abort (mach);
    }

  if ((rc = moderator_message_get_part (mach, msg, 2, &orig)))
    mu_sieve_abort (mach);

  rc = moderator_filter_message (mach, tags, orig, &discard);
  mu_message_unref (orig);
  if (rc)
    mu_sieve_abort (mach);

  if (discard && !mu_sieve_is_dry_run (mach))
    {
      mu_message_t request;
      char *from = NULL;
      mu_sieve_value_t *arg;
      
      if ((rc = moderator_message_get_part (mach, msg, 3, &request)))
	{
	  mu_sieve_error (mach, _("cannot get message part #3: %s"),
			  mu_strerror (rc));
	  mu_sieve_abort (mach);
	}

      if (mu_sieve_tag_lookup (tags, "address", &arg))
	from = arg->v.string;
      
      if (moderator_discard_message (mach, request, from))
	discard = 0;
      else
	{
	  if (!mu_sieve_tag_lookup (tags, "keep", NULL))
	    {
	      mu_attribute_t attr = 0;

	      if (mu_message_get_attribute (msg, &attr) == 0)
		mu_attribute_set_deleted (attr);
	    }
	  else
	    discard = 0;
	}
      mu_message_unref (request);
    }

  mu_sieve_log_action (mach, "MODERATOR", 
		       discard ? _("discarding message") :
		       _("keeping message"));
  return 0;
}


/* Initialization */
   
/* Required arguments: */
static mu_sieve_data_type moderator_req_args[] = {
  SVT_VOID
};

/* Tagged arguments: */
static mu_sieve_tag_def_t moderator_tags[] = {
  { "keep", SVT_VOID },
  { "address", SVT_STRING },
  { "source", SVT_STRING },
  { NULL }
};

static mu_sieve_tag_group_t moderator_tag_groups[] = {
  { moderator_tags, NULL },
  { NULL }
};


/* Initialization function. */
int
SIEVE_EXPORT(moderator,init) (mu_sieve_machine_t mach)
{
  return mu_sieve_register_action (mach, "moderator", moderator_action,
				   moderator_req_args,
				   moderator_tag_groups, 1);
}
   
