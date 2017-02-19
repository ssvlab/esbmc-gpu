/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008,
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#ifdef HAVE_MALLOC_H
# include <malloc.h>
#endif

#include <mailutils/mailutils.h>
#include "mailutils/libargp.h"

static int messages_count (const char *);

const char *program_version = "messages (" PACKAGE_STRING ")";
static char doc[] = N_("GNU messages -- count the number of messages in a mailbox");
static char args_doc[] = N_("[mailbox...]");

static struct argp_option options[] = {
  { NULL,         0, NULL,  0,
    /* TRANSLATORS: 'messages' is a program name. Do not translate it! */
    N_("messages specific switches:"), 0},
  {"quiet",	'q',	NULL,	0,	N_("only display number of messages")},
  {"silent",	's',	NULL,	OPTION_ALIAS, NULL },
  { 0 }
};

static const char *argp_capa[] = {
  "common",
  "debug",
  "license",
  "mailbox",
  "locking",
  NULL
};

struct arguments
{
  int argc;
  char **argv;
};

/* are we loud or quiet? */
static int silent = 0;

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  struct arguments *args = state->input;
  switch (key)
    {
    case 'q':
    case 's':
      silent = 1;
      break;
      
    case ARGP_KEY_ARG:
      args->argv = realloc (args->argv,
			    sizeof (char *) * (state->arg_num + 2));
      args->argv[state->arg_num] = arg;
      args->argv[state->arg_num + 1] = NULL;
      args->argc++;
      break;
      
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {
  options,
  parse_opt,
  args_doc,
  doc,
  NULL,
  NULL, NULL
};

int
main (int argc, char **argv)
{
  int i = 1;
  int err = 0;
  struct arguments args = {0, NULL};

  /* Native Language Support */
  MU_APP_INIT_NLS ();

  /* register the formats.  */
  mu_register_all_mbox_formats ();

#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif
  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, argp_capa, NULL, argc, argv, 0, NULL, &args))
    exit (1);

  if (args.argc < 1 && messages_count (NULL) < 0)
    err = 1;
  else if (args.argc >= 1)
    {
      for (i = 0; i < args.argc; i++)
	{
	  if (messages_count (args.argv[i]) < 0)
	    err = 1;
	}
    }

  return err;
}

static int
messages_count (const char *box)
{
  mu_mailbox_t mbox;
  mu_url_t url = NULL;
  size_t count;
  int status = 0;

  status =  mu_mailbox_create_default (&mbox, box);
  if (status != 0)
    {
      if (box)
	mu_error (_("could not create mailbox `%s': %s"),
		  box, mu_strerror (status));
      else
	mu_error (_("could not create default mailbox: %s"),
		  mu_strerror (status));
      return -1;
    }

  mu_mailbox_get_url (mbox, &url);
  box = mu_url_to_string (url);

  status =  mu_mailbox_open (mbox, MU_STREAM_READ);
  if (status != 0)
    {
      mu_error (_("could not open mailbox `%s': %s"),
		box, mu_strerror (status));
      return -1;
    }

  status = mu_mailbox_messages_count (mbox, &count);
  if (status != 0)
    {
      mu_error (_("could not count messages in mailbox `%s': %s"),
		box, mu_strerror (status));
      return -1;
    }

  if (silent)
    printf ("%lu\n", (unsigned long) count);
  else
    printf (_("Number of messages in %s: %lu\n"), box, (unsigned long) count);

  status = mu_mailbox_close (mbox);
  if (status != 0)
    {
      mu_error (_("could not close `%s': %s"),
		box, mu_strerror (status));
      return -1;
    }

  mu_mailbox_destroy (&mbox);
  return count;
}
