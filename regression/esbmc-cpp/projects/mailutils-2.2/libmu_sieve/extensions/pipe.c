/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

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

/* Syntax: pipe [:envelope] <program: string>

   The pipe action executes a shell command specified by its
   argument and pipes the entire message to its standard input.
   The envelope of the message is included, if the :envelope tag is given.
   
   Notes/FIXME: 1. it would be nice to implement meta-variables in
                <program call> which would expand to various
		items from the message being handled.
		2. :mime tag could be useful too.
*/

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <signal.h>
#include <regex.h>
#include <mailutils/sieve.h>

#define ASSERT(expr, diag, ec)                                                \
 if (!(expr))                                                                 \
   {                                                                          \
     if (ec)                                                                  \
       mu_sieve_error (mach, "%lu: %s: %s",                                   \
	 	       (unsigned long) mu_sieve_get_message_num (mach),	      \
		       diag,                                                  \
		       mu_strerror (ec));                                     \
     else                                                                     \
       mu_sieve_error (mach, "%lu: %s",                                       \
	 	       (unsigned long) mu_sieve_get_message_num (mach),	      \
		       diag);                                                 \
     mu_sieve_abort (mach);                                                   \
   }
    
#define ASSERT2(expr, diag, arg, ec)                                          \
 if (!(expr))                                                                 \
   {                                                                          \
     if (ec)                                                                  \
       mu_sieve_error (mach, "%lu: `%s': %s: %s",                             \
	 	       (unsigned long) mu_sieve_get_message_num (mach),	      \
		       arg,                                                   \
		       diag,                                                  \
		       mu_strerror (ec));                                     \
     else                                                                     \
       mu_sieve_error (mach, "%lu: `%s': %s",                                 \
		       (unsigned long) mu_sieve_get_message_num (mach),	      \
		       arg,                                                   \
		       diag);                                                 \
     mu_sieve_abort (mach);                                                   \
   }

int
sieve_action_pipe (mu_sieve_machine_t mach, mu_list_t args, mu_list_t tags)
{
  int rc;
  mu_message_t msg;
  mu_sieve_value_t *val;
  char *cmd;
  mu_stream_t mstr, pstr;
  char buf[512];
  size_t n;
  mu_envelope_t env;
  
  val = mu_sieve_value_get (args, 0);
  ASSERT (val, _("cannot get command!"), 0);
  cmd = val->v.string;

  mu_sieve_log_action (mach, "PIPE", NULL);
  if (mu_sieve_get_debug_level (mach) & MU_SIEVE_DEBUG_TRACE)
    {
      mu_sieve_locus_t locus;
      mu_sieve_get_locus (mach, &locus);
      mu_sieve_debug (mach, "%s:%lu: PIPE\n",
		      locus.source_file,
		      (unsigned long) locus.source_line);
    }

  if (mu_sieve_is_dry_run (mach))
    return 0;

  msg = mu_sieve_get_message (mach);
  mu_message_get_envelope (msg, &env);
  
  rc = mu_message_get_stream (msg, &mstr);
  ASSERT (rc == 0, _("cannot get message stream"), rc);
  
  rc = mu_prog_stream_create (&pstr, cmd, MU_STREAM_WRITE);
  ASSERT2 (rc == 0, _("cannot create command stream"), cmd, rc);

  rc = mu_stream_open (pstr);
  ASSERT2 (rc == 0, _("cannot open command stream"), cmd, rc);

  if (mu_sieve_tag_lookup (tags, "envelope", &val))
    {
      char *p;

      rc = mu_envelope_aget_sender (env, &p);
      ASSERT (rc == 0, _("cannot get envelope sender"), rc);
      rc = mu_stream_sequential_write (pstr, "From ", 5);
      ASSERT (rc == 0, _("stream write failed"), rc);
      mu_stream_sequential_write (pstr, p, strlen (p));
      free (p);
      rc = mu_stream_sequential_write (pstr, " ", 1);
      ASSERT (rc == 0, _("stream write failed"), rc);
      rc = mu_envelope_aget_date (env, &p);
      ASSERT (rc == 0, _("cannot get envelope date"), rc);
      rc = mu_stream_sequential_write (pstr, p, strlen (p));
      ASSERT (rc == 0, _("stream write failed"), rc);
      free (p);
      rc = mu_stream_sequential_write (pstr, "\n", 1);
      ASSERT (rc == 0, _("stream write failed"), rc);
    }
  
  mu_stream_seek (mstr, 0, SEEK_SET);
  while (rc == 0
	 && mu_stream_sequential_read (mstr, buf, sizeof buf, &n) == 0
	 && n > 0)
    rc = mu_stream_sequential_write (pstr, buf, n);

  mu_stream_close (pstr);
  mu_stream_destroy (&pstr, mu_stream_get_owner (pstr));


  ASSERT2 (rc == 0, _("command failed"), cmd, rc);
  
  return 0;
}

/* Tagged arguments: */
static mu_sieve_tag_def_t pipe_tags[] = {
  { "envelope", SVT_VOID },
  { NULL }
};
  
static mu_sieve_tag_group_t pipe_tag_groups[] = {
  { pipe_tags, NULL }, 
  { NULL }
};

/* Required arguments: */
static mu_sieve_data_type pipe_args[] = {
  SVT_STRING,			/* program call */
  SVT_VOID
};

int
SIEVE_EXPORT (pipe, init) (mu_sieve_machine_t mach)
{
  return mu_sieve_register_action (mach, "pipe", sieve_action_pipe,
				   pipe_args, pipe_tag_groups, 1);
}

