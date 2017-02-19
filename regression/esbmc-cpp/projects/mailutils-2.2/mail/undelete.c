/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2007, 2010 Free Software
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
 * u[ndelete] [msglist]
 */

static int
undelete0 (msgset_t *mspec, mu_message_t msg, void *data)
{
  mu_attribute_t attr;

  mu_message_get_attribute (msg, &attr);
  mu_attribute_unset_deleted (attr);
  mu_attribute_unset_userflag (attr, MAIL_ATTRIBUTE_PRESERVED);
  mu_attribute_unset_userflag (attr, MAIL_ATTRIBUTE_MBOXED);
  util_mark_read (msg);
  cond_page_invalidate (mspec->msg_part[0]);

  return 0;
}

int
mail_undelete (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_ALL, undelete0, NULL);
}

