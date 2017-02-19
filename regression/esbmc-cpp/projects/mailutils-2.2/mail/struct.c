/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils.  If not, see <http://www.gnu.org/licenses/>. */

#include "mail.h"

#define PART_WIDTH 8

static int
show_part (struct mime_descend_closure *closure, void *data)
{
  size_t width;
  size_t size = 0;
  
  width = fprint_msgset (ofile, closure->msgset);
  for (; width < 5; width++)
    fputc (' ', ofile);
    
  fprintf (ofile, " %-25s", closure->type);

  mu_message_size (closure->message, &size);
  if (size < 1024)
    fprintf (ofile, " %4lu", (unsigned long) size);
  else if (size < 1024*1024)
    fprintf (ofile, "%4luK", (unsigned long) size / 1024);
  else
    fprintf (ofile, "%4luM", (unsigned long) size / 1024 / 1024);

  fprintf (ofile, "\n");
  return 0;
}

static int
show_struct (msgset_t *msgset, mu_message_t msg, void *data)
{
  struct mime_descend_closure mclos;
  
  mclos.hints = 0;
  mclos.msgset = msgset;
  mclos.message = msg;
  mclos.type = NULL;
  mclos.encoding = NULL;
  mclos.parent = NULL;
  
  mime_descend (&mclos, show_part, NULL);

    /* Mark enclosing message as read */
  if (mu_mailbox_get_message (mbox, msgset->msg_part[0], &msg) == 0)
    util_mark_read (msg);

  return 0;
}

int
mail_struct (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_NODELETED|MSG_SILENT,
			   show_struct, NULL);
}
