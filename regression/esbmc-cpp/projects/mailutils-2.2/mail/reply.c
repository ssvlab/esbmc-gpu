/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2005, 2007, 2009, 2010 Free Software
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

void
make_in_reply_to (compose_env_t *env, mu_message_t msg)
{
  char *value = NULL;

  mu_rfc2822_in_reply_to (msg, &value);
  compose_header_set (env, MU_HEADER_IN_REPLY_TO, value,
		      COMPOSE_REPLACE);
  free (value);
}

void
make_references (compose_env_t *env, mu_message_t msg)
{
  char *value = NULL;

  mu_rfc2822_references (msg, &value);
  compose_header_set (env, MU_HEADER_REFERENCES, value, COMPOSE_REPLACE);
  free (value);
}
  
/*
 * r[eply] [msglist] -- GNU extension
 * r[espond] [msglist] -- GNU extension
 * R[eply] [msglist]
 * R[espond] [msglist]
 */

int
reply0 (msgset_t *mspec, mu_message_t msg, void *data)
{
  mu_header_t hdr;
  compose_env_t env;
  int status;
  char *str;

  set_cursor (mspec->msg_part[0]);
  
  compose_init (&env);

  mu_message_get_header (msg, &hdr);

  compose_header_set (&env, MU_HEADER_TO,
		      util_get_sender (mspec->msg_part[0], 0),
		      COMPOSE_SINGLE_LINE);

  if (*(int*) data) /* reply starts with a lowercase */
    {
      /* Add all recepients of the originate letter */

      mu_address_t addr = NULL;
      size_t i, count = 0;

      if (mu_header_aget_value (hdr, MU_HEADER_TO, &str) == 0)
	{
	  mu_address_create (&addr, str);
	  free (str);
	  mu_address_get_count (addr, &count);
	}

      /* Make sure we do not include our alternate names */
      for (i = 1; i <= count; i++)
	{
	  const char *email;
	  if (mu_address_sget_email (addr, i, &email) || email == NULL)
	    continue;
	  if ((mailvar_get (NULL, "metoo", mailvar_type_boolean, 0) == 0)
	      || !mail_is_my_name (email))
	    compose_header_set (&env, MU_HEADER_TO,
				email,
				COMPOSE_SINGLE_LINE);
	}
      
      mu_address_destroy (&addr);

      /* Finally, add any Ccs */
      if (mu_header_aget_value (hdr, MU_HEADER_CC, &str) == 0)
	compose_header_set (&env, MU_HEADER_TO, str, COMPOSE_SINGLE_LINE);
    }

  if (mu_header_aget_value (hdr, MU_HEADER_SUBJECT, &str) == 0)
    {
      char *p = NULL;
      
      if (mu_unre_subject (str, NULL))
	util_strcat (&p, util_reply_prefix ());
      util_strcat (&p, str);
      free (str);
      compose_header_set (&env, MU_HEADER_SUBJECT, p, COMPOSE_REPLACE);
      free (p);
    }
  else
    compose_header_set (&env, MU_HEADER_SUBJECT, "", COMPOSE_REPLACE);

  fprintf (ofile, "To: %s\n",
	   compose_header_get (&env, MU_HEADER_TO, ""));
  str = compose_header_get (&env, MU_HEADER_CC, NULL);
  if (str)
    fprintf (ofile, "Cc: %s\n", str);
  fprintf (ofile, "Subject: %s\n\n",
	   compose_header_get (&env, MU_HEADER_SUBJECT, ""));
  
  make_in_reply_to (&env, msg);
  make_references (&env, msg);
  status = mail_send0 (&env,
		       mailvar_get (NULL, "byname", mailvar_type_boolean, 0)
		                     == 0);
  compose_destroy (&env);

  return status;
}

int
mail_reply (int argc, char **argv)
{
  int lower = mu_islower (argv[0][0]);
  if (mailvar_get (NULL, "flipr", mailvar_type_boolean, 0) == 0)
    lower = !lower;
  return util_foreach_msg (argc, argv, MSG_NODELETED, reply0, &lower);
}

