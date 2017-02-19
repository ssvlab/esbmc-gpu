/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2003, 2005, 2007, 2010 Free Software
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
 * pi[pe] [[msglist] command]
 * | [[msglist] command]
 */

int
mail_pipe (int argc, char **argv)
{
  mu_message_t msg;
  mu_stream_t stream;
  char *cmd;
  FILE *tube;
  msgset_t *list, *mp;
  char buffer[512];
  off_t off = 0;
  size_t n = 0;

  if (argc > 2)
    cmd = argv[--argc];
  else if (mailvar_get (&cmd, "cmd", mailvar_type_string, 1))
    return 1;

  if (msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &list))
      return 1;

  tube = popen (cmd, "w");

  for (mp = list; mp; mp = mp->next)
    {
      if (util_get_message (mbox, mp->msg_part[0], &msg) == 0)
	{
	  mu_message_get_stream (msg, &stream);
	  off = 0;
	  while (mu_stream_read (stream, buffer, sizeof (buffer) - 1, off,
			      &n) == 0 && n != 0)
	    {
	      buffer[n] = '\0';
	      fprintf (tube, "%s", buffer);
	      off += n;
	    }
	  if (mailvar_get (NULL, "page", mailvar_type_boolean, 0) == 0)
	    fprintf (tube, "\f\n");
	}
      util_mark_read (msg);
    }

  msgset_free (list);
  pclose (tube);
  return 0;
}
