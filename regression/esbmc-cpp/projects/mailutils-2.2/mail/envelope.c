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

/*
 * env[elope] [msglist]
 */

int
print_envelope (msgset_t *mspec, mu_message_t msg, void *data)
{
  int status;
  mu_envelope_t env = NULL;
  const char *sender = NULL, *date = NULL;

  status = mu_message_get_envelope (msg, &env);
  if (status)
    {
      mu_error (_("%lu: Cannot get envelope"),
		(unsigned long) mspec->msg_part[0]);
    }
  else
    {
      mu_envelope_sget_sender (env, &sender);
      mu_envelope_sget_date (env, &date);
      if (data)
	fprintf (ofile, "%s ", (char*) data);
      fprintf (ofile, "%s %s", sender, date);
    }
  return 0;
}

int
mail_envelope (int argc, char **argv)
{
  return util_foreach_msg (argc, argv, MSG_NODELETED|MSG_SILENT,
			   print_envelope, NULL);
}
