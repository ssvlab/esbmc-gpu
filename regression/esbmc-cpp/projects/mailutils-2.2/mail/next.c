/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2005, 2007, 2010 Free Software
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
 * n[ext] [message]
 * +
 */

int
mail_next (int argc, char **argv)
{
  size_t n;
  mu_message_t msg;
  
  if (argc < 2)
    {
      int rc;
      mu_attribute_t attr = NULL;
      
      n = get_cursor ();
      if (n == 0 || util_get_message (mbox, n, &msg))
	{
	  util_error (_("No applicable message"));
	  return 1;
	}

      mu_message_get_attribute (msg, &attr);
      if (!mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SHOWN))
	{
	  util_do_command ("print");
	  return 0;
	}
      
      rc = 1;
      while (++n <= total)
	{
	  if (util_isdeleted (n))
	    continue;
	  rc = util_get_message (mbox, n, &msg);
	  if (rc == 0)
	    break;
	}

      if (rc)
	{
	  util_error (_("No applicable message"));
	  return 1;
	}
    }
  else
    {
      msgset_t *list = NULL;
      int rc = msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &list);
      if (!rc)
	{
	  n = list->msg_part[0];
	  msgset_free (list);
	  if (util_get_message (mbox, n, &msg))
	    return 1;
	}
      else
	{
	  util_error (_("No applicable message"));
	  return 1;
	}
    }
  set_cursor (n);
  util_do_command ("print");
  return 0;
}
