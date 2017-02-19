/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software Foundation,
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

/*
 * ve[rsion]
 */

static const char *with_defs[] =
{
#ifdef WITH_PTHREAD
  "PTHREAD",
#endif
#ifdef WITH_BDB2
  "BDB2",
#endif
#ifdef WITH_READLINE
  "READLINE",
#endif
  NULL
};


int
mail_version (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED)
{ 
  fprintf (ofile, "%s", program_version);
  if (with_defs[0] != NULL)
    {
      int i;
      fprintf (ofile, " (");
      for (i = 0; with_defs[i]; i++)
	fprintf (ofile, " %s", with_defs[i]);
      fprintf (ofile, " )");
    }
  fprintf (ofile, "\n");

  return 0;
}
