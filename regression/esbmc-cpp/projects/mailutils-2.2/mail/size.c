/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2009, 2010 Free Software
   Foundation, Inc.

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
 * si[ze] [msglist]
 */

static int
size0 (msgset_t *mspec, mu_message_t msg, void *data)
{
  size_t size = 0, lines = 0;

  mu_message_size (msg, &size);
  mu_message_lines (msg, &lines);
  
  fprintf (ofile, "%c%2lu %3lu/%-5lu\n",
	   is_current_message (mspec->msg_part[0]) ? '>' : ' ',
	   (unsigned long) mspec->msg_part[0],
	   (unsigned long) lines,
	   (unsigned long) size);
  return 0;
}

int
mail_size (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_ALL, size0, NULL);
}


