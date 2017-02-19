/* This file is part of GNU Mailutils.
   Copyright (C) 1998, 2001, 2002, 2005, 2007, 2009, 2010 Free Software
   Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; see the file COPYING.  If not, write
   to the Free Software Foundation, Inc., 51 Franklin Street,
   Fifth Floor, Boston, MA 02110-1301 USA. */

#include "comsat.h"

static int
print_and_free_acl (void *item, void *data)
{
  FILE *outfile = data;
  char **argv = item;
  
  fprintf (outfile, "  %s from %s;\n", argv[1], argv[2]);
  mu_argv_free (argv);
  return 0;
}
    
void
convert_config (const char *config_file, FILE *outfile)
{
  FILE *fp;
  int line;
  char buf[128];
  char *ptr;
  mu_list_t aclist = NULL;
  
  if (!config_file)
    return;

  fp = fopen (config_file, "r");
  if (!fp)
    {
      mu_error (_("cannot open config file %s: %s"), config_file,
		mu_strerror (errno));
      return;
    }

  fprintf (outfile,
	   "# Configuration file for GNU comsatd, converted from %s\n",
	   config_file);
  fprintf (outfile,
	   "# Copy it to the comsatd configuration file\n");
  fprintf (outfile,
	   "# or to %s/mailutils.rc, in section `program %s'\n\n",
	   SYSCONFDIR, mu_program_name);

  line = 0;
  while ((ptr = fgets (buf, sizeof buf, fp)))
    {
      int len;
      int argc;
      char **argv;

      line++;
      len = strlen (ptr);
      if (len > 0 && ptr[len-1] == '\n')
	ptr[--len] = 0;

      while (*ptr && mu_isblank (*ptr))
	ptr++;
      if (!*ptr || *ptr == '#')
	{
	  fprintf (outfile, "%s\n", ptr);
	  continue;
	}

      mu_argcv_get (ptr, "", NULL, &argc, &argv);
      if (argc < 2)
	{
	  mu_error (_("%s:%d: too few fields"), config_file, line);
	  mu_argcv_free (argc, argv);
	  continue;
	}

      if (strcmp (argv[0], "acl") == 0)
	{
	  if (!aclist)
	    mu_list_create (&aclist);
	  mu_list_append (aclist, argv);
	}
      else
	{
	  mu_argcv_free (argc, argv);
	  fprintf (outfile, "%s;\n", ptr);
	}
    }
  fclose (fp);

  if (aclist)
    {
      fprintf (outfile, "acl {\n");
      mu_list_do (aclist, print_and_free_acl, outfile);
      fprintf (outfile, "};\n");
      mu_list_destroy (&aclist);
    }
}

