/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2005, 2007, 2008, 2009, 2010 Free Software Foundation,
   Inc.

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

int count_only;
char *sender_option;
char *mailbox_name;

const char *program_version = "from (" PACKAGE_STRING ")";
static char doc[] = N_("GNU from -- display from and subject.");

static struct argp_option options[] = {
  {"count",  'c', NULL,   0, N_("just print a count of messages and exit")},
  {"sender", 's', N_("ADDRESS"), 0,
   N_("print only mail from addresses containing the supplied string") },
  {"file",   'f', N_("FILE"), 0,
   N_("read mail from FILE") },
  {"debug",  'd', NULL,   0, N_("enable debugging output"), 0},
  {0, 0, 0, 0}
};

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'c':
      count_only = 1;
      break;
      
    case 's':
      sender_option = arg;
      break;
      
    case 'f':
      mailbox_name = arg;
      break;
      
    case 'd':
      frm_debug++;
      break;

    default: 
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {
  options,
  parse_opt,
  N_("[OPTIONS] [USER]"),
  doc,
};

static const char *capa[] = {
  "common",
  "debug",
  "license",
  "mailbox",
  "locking",
  NULL
};

static int
from_select (size_t index, mu_message_t msg)
{
  if (count_only)
    return 0;

  if (sender_option)
    {
      int rc = 0;
      mu_header_t hdr = NULL;
      char *sender;
      mu_message_get_header (msg, &hdr);

      if (mu_header_aget_value_unfold (hdr, MU_HEADER_FROM, &sender) == 0)
	{
	  if (strstr (sender, sender_option))
	    rc = 1;
	  free (sender);
	}
      
      return rc;
    }
  
  return 1;
}

int
main (int argc, char **argv)
{
  int c;
  size_t total;
  
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  /* register the formats.  */
  mu_register_all_mbox_formats ();
#ifdef WITH_TLS
  mu_gocs_register ("tls", mu_tls_module_init);
#endif

  mu_argp_init (program_version, NULL);
  if (mu_app_init (&argp, capa, NULL, argc, argv, 0, &c, NULL))
    exit (1);

  if (argc - c > 1)
    {
      mu_error (_("too many arguments"));
      exit (1);
    }
  else if (argc - c > 0)
    {
      if (mailbox_name)
	{
	  mu_error (_("both --from option and user name are specified"));
	  exit (1);
	}

      mailbox_name = xmalloc (strlen (argv[c]) + 2);
      mailbox_name[0] = '%';
      strcpy (mailbox_name + 1, argv[c]);
    }

  init_output (0);
  
  frm_scan (mailbox_name, from_select, &total);

  if (count_only)
    {
      printf (ngettext ("There is %lu message in your incoming mailbox.\n",
			"There are %lu messages in your incoming mailbox.\n",
			total),
	      (unsigned long) total);
    }
  return 0;
}
  
