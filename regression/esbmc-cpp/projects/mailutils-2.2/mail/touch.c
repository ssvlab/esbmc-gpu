/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2005, 2007, 2010 Free Software Foundation,
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

/* tou[ch] [msglist]
   
   Touch the specified messages. If any message in msglist is not
   specifically deleted nor saved in a file, it shall be placed in the
   mbox upon normal termination. */

static int
touch0 (msgset_t *mspec, mu_message_t msg, void *data)
{
  mu_attribute_t attr = NULL;
  
  mu_message_get_attribute (msg, &attr);
  mu_attribute_set_userflag (attr, MAIL_ATTRIBUTE_TOUCHED);
  
  set_cursor (mspec->msg_part[0]);
  return 0;
}

int
mail_touch (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_NODELETED, touch0, NULL);
  return mail_mbox (argc, argv);
}
