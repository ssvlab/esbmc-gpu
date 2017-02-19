/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2005, 2007, 2010 Free Software Foundation, Inc.

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
 * cd [directory]
 * ch[dir] [directory]
 */

int
mail_cd (int argc, char **argv)
{
  char *dir;
  
  if (argc > 2)
    return 1;
  else if (argc == 2)
    dir = argv[1];
  else 
    dir = getenv ("HOME");

  if (chdir (dir))
    {
      mu_diag_funcall (MU_DIAG_ERROR, "chdir", dir, errno);
      return 1;
    }
  return 0;
}
