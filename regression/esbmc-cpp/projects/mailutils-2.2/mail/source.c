/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2007, 2010 Free Software Foundation,
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

#include "mail.h"

static char *
source_readline (void *closure, int cont MU_ARG_UNUSED)
{
  FILE *fp = closure;
  size_t s = 0;
  char *buf = NULL;
  mu_debug_t debug;
  struct mu_debug_locus locus;
  
  if (getline (&buf, &s, fp) >= 0)
    {
      mu_rtrim_class (buf, MU_CTYPE_SPACE);

      mu_diag_get_debug (&debug);
      mu_debug_get_locus (debug, &locus);
      mu_debug_set_locus (debug, locus.file, locus.line + 1);
      return buf;
    }
  
  return NULL;
}
  
/*
 * so[urce] file
 */

int
mail_source (int argc, char **argv)
{
  FILE *fp;
  int save_term;
  mu_debug_t debug;
  
  if (argc != 2)
    {
      /* TRANSLATORS: 'source' is a command name. Do not translate it! */
      util_error (_("source requires a single argument"));
      return 1;
    }
  
  fp = fopen (argv[1], "r");
  if (!fp)
    {
      if (errno != ENOENT)
	util_error(_("Cannot open `%s': %s"), argv[1], strerror(errno));
      return 1;
    }

  save_term = interactive;
  interactive = 0;
  mu_diag_get_debug (&debug);
  mu_debug_set_locus (debug, argv[1], 0);
  mail_mainloop (source_readline, fp, 0);
  interactive = save_term;
  mu_debug_set_locus (debug, NULL, 0);
  fclose (fp);
  return 0;
}
