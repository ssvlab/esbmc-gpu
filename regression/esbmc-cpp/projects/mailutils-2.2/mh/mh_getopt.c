/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2010
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

/* Parse traditional MH options. */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mh_getopt.h>
#include <mu_asprintf.h>

static int mh_optind = 1;
static char *mh_optarg;
static char *mh_optptr;

int
mh_getopt (int argc, char **argv, struct mh_option *mh_opt, const char *doc)
{
  struct mh_option *p;
  int optlen;
  
  if (mh_optind >= argc || argv[mh_optind] == NULL)
    return EOF;
  mh_optptr = argv[mh_optind];

  if (mh_optptr[0] == '+')
    {
      mh_optarg = mh_optptr + 1;
      mh_optind++;
      return '+';
    }
  
  if (mh_optptr[0] != '-' || mh_optptr[1] == '-')
    {
      mh_optind++;
      return 0;
    }

  if (strcmp (mh_optptr, "-version") == 0)
    asprintf (&argv[mh_optind], "--version");
  else
    {
      optlen = strlen (mh_optptr+1);
      for (p = mh_opt; p->opt; p++)
	{
	  if ((p->match_len <= optlen
	       && memcmp (mh_optptr+1, p->opt, optlen) == 0)
	      || (p->flags == MH_OPT_BOOL
		  && optlen > 2
		  && memcmp (mh_optptr+1, "no", 2) == 0
		  && strlen (p->opt) >= optlen-2
		  && memcmp (mh_optptr+3, p->opt, optlen-2) == 0))
	    break;
	}
      
      if (p->opt)
	{
	  char *longopt = p->opt;
	  switch (p->flags)
	    {
	    case MH_OPT_BOOL:
	      if (memcmp (mh_optptr+1, "no", 2) == 0)
		mh_optarg = "no";
	      else
		mh_optarg = "yes";
	      asprintf (&argv[mh_optind], "--%s=%s", longopt, mh_optarg);
	      break;
	      
	    case MH_OPT_ARG:
	      asprintf (&argv[mh_optind], "--%s", longopt);
	      mh_optarg = argv[++mh_optind];
	      break;
	      
	    default:
	      asprintf (&argv[mh_optind], "--%s", longopt);
	      mh_optarg = NULL;
	    }
	  mh_optind++;
	  return 1;
	}
      else if (!strcmp (mh_optptr+1, "help"))
	{
	  mh_help (mh_opt, doc);
	  exit (1);
	}
      else
	mh_optind++;
    }
  return '?';
}

void
mh_argv_preproc (int argc, char **argv, struct mh_argp_data *data)
{
  mh_optind = 1;
  while (mh_getopt (argc, argv, data->mh_option, data->doc) != EOF)
    ;
}

void
mh_help (struct mh_option *mh_opt, const char *doc)
{
  struct mh_option *p;

  printf (_("Compatibility syntax:\n"));
  printf (_("%s [switches] %s\n"), mu_program_name, doc);
  printf (_("  switches are:\n"));
  
  for (p = mh_opt; p->opt; p++)
    {
      int len = strlen (p->opt);
      
      printf ("  -");
      if (p->flags == MH_OPT_BOOL)
	printf ("[no]");
      if (len > p->match_len)
	printf ("(%*.*s)%s",
		p->match_len, p->match_len, p->opt,
		p->opt + p->match_len);
      else
	printf ("%s", p->opt);
      
      if (p->flags == MH_OPT_ARG)
	printf (" %s", p->arg);
      printf ("\n");
    }
  printf ("  -help\n");
  printf ("  -version\n");
  printf (_("\nPlease use GNU long options instead.\n"
            "Run %s --help for more info on these.\n"),
            mu_program_name);
}

void
mh_opt_notimpl (const char *name)
{
  mu_error (_("option is not yet implemented: %s"), name);
  exit (1);
}

void
mh_opt_notimpl_warning (const char *name)
{
  mu_error (_("ignoring not implemented option %s"), name);
}
