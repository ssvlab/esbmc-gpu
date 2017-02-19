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

/* ta[g] [msglist] */
/* unt[ag] [msglist] */

static int
tag_message (mu_message_t mesg, msgset_t *msgset MU_ARG_UNUSED, void *arg)
{
  mu_attribute_t attr;
  int *action = arg;

  mu_message_get_attribute (mesg, &attr);
  if (*action)
    mu_attribute_set_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
  else
    mu_attribute_unset_userflag (attr, MAIL_ATTRIBUTE_TAGGED);
  return 0;
}

int
mail_tag (int argc, char **argv)
{
  msgset_t *msgset;
  int action = argv[0][0] != 'u';

  if (msgset_parse (argc, argv, MSG_NODELETED|MSG_SILENT, &msgset))
    return 1;

  util_msgset_iterate (msgset, tag_message, (void *)&action);

  msgset_free (msgset);
  return 0;
}
