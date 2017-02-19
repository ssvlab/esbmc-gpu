/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008,
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

/* fmtcheck */

#include <mh.h>

const char *program_version = "fmtcheck (" PACKAGE_STRING ")";
static char doc[] = N_("GNU MH fmtcheck")"\v"
N_("Use -help to obtain the list of traditional MH options.");
static char args_doc[] = "";

/* GNU options */
static struct argp_option options[] = {
  {"form",    ARG_FORM, N_("FILE"),   0,
   N_("read format from given file")},
  {"format",  ARG_FORMAT, N_("FORMAT"), 0,
   N_("use this format string")},
  {"dump",    ARG_DUMP, NULL,     0,
   N_("dump the listing of compiled format code")},
  { "debug",  ARG_DEBUG, NULL,     0,
    N_("enable parser debugging output"),},
  {"license", ARG_LICENSE, 0,      0,
   N_("display software license"), -1},

  { 0 }
};

/* Traditional MH options */
struct mh_option mh_option[] = {
  {"form",    4,  MH_OPT_ARG, "formatfile"},
  {"format",  5,  MH_OPT_ARG, "string"},
  { 0 }
};

char *format_str;
static mh_format_t format;

typedef int (*action_fp) (void);

static int
action_dump ()
{
  if (!format_str)
    {
      mu_error (_("Format string not specified"));
      return 1;
    }
  mh_format_dump (&format);
  return 0;
}

static action_fp action = action_dump;

static error_t
opt_handler (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case ARG_FORM:
      mh_read_formfile (arg, &format_str);
      break;

    case ARG_FORMAT:
      format_str = arg;
      break;

    case ARG_DUMP:
      action = action_dump;
      break;

    case ARG_DEBUG:
      mh_format_debug (1);
      break;
      
    case ARG_LICENSE:
      mh_license (argp_program_version);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

int
main (int argc, char **argv)
{
  /* Native Language Support */
  MU_APP_INIT_NLS ();

  mh_argp_init (program_version);
  mh_argp_parse (&argc, &argv, 0, options, mh_option, args_doc, doc,
		 opt_handler, NULL, NULL);

  if (format_str && mh_format_parse (format_str, &format))
    {
      mu_error (_("Bad format string"));
      exit (1);
    }
  return (*action) ();
}
