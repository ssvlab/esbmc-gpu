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
 * q[uit]
 * <EOF>
 */

int
mail_quit (int argc MU_ARG_UNUSED, char **argv MU_ARG_UNUSED)
{
  if (mail_mbox_close ())
    return 1;
  exit (0);
}

int
mail_mbox_close ()
{
  mu_url_t url = NULL;
  size_t held_count = 0;

  if (!mbox)
    return 0;

  if (mailvar_get (NULL, "readonly", mailvar_type_boolean, 0))
    {
      if (mail_mbox_commit ())
	return 1;

      mu_mailbox_flush (mbox, 1);
    }
  
  mu_mailbox_get_url (mbox, &url);
  mu_mailbox_messages_count (mbox, &held_count);
  fprintf (ofile, 
           ngettext ("Held %d message in %s\n",
                     "Held %d messages in %s\n",
                     held_count),
           held_count, util_url_to_string (url));
  mu_mailbox_close (mbox);
  mu_mailbox_destroy (&mbox);
  return 0;
}

int
mail_mbox_commit ()
{
  unsigned int i;
  mu_mailbox_t dest_mbox = NULL;
  int saved_count = 0;
  mu_message_t msg;
  mu_attribute_t attr;
  int keepsave = mailvar_get (NULL, "keepsave", mailvar_type_boolean, 0) == 0;
  int hold = mailvar_get (NULL, "hold", mailvar_type_boolean, 0) == 0;
  mu_url_t url;
  int is_user_mbox;

  mu_mailbox_get_url (mbox, &url);
  is_user_mbox = strcmp (util_url_to_string (url), getenv ("MBOX")) == 0;

  {
    mu_mailbox_t mb;
    mu_url_t u;
    mu_mailbox_create_default (&mb, NULL);
    mu_mailbox_get_url (mb, &u);
    if (strcmp (mu_url_to_string (u), mu_url_to_string (url)) != 0)
      {
	/* The mailbox we are closing is not a system one (%). Raise
	   hold flag */
	hold = 1;
	keepsave = 1;
      }
    mu_mailbox_destroy (&mb);
  }

  for (i = 1; i <= total; i++)
    {
      if (util_get_message (mbox, i, &msg))
	return 1;

      mu_message_get_attribute (msg, &attr);

      if (!is_user_mbox
	  && (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_MBOXED)
	      || (!hold
		  && !mu_attribute_is_deleted (attr)
		  && !mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_PRESERVED)
		  && ((mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SAVED)
		       && keepsave)
		      || (!mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SAVED)
			  && (mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SHOWN)
			      || mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_TOUCHED)))))))
	{
	  int status;
	  
	  if (!dest_mbox)
	    {
	      char *name = getenv ("MBOX");
	       
	      if ((status = mu_mailbox_create_default (&dest_mbox, name)) != 0)
		{
		  util_error (_("Cannot create mailbox %s: %s"), name,
                              mu_strerror (status));
		  return 1;
		}
              if ((status = mu_mailbox_open (dest_mbox,
			   	          MU_STREAM_WRITE | MU_STREAM_CREAT))!=0)
		{
		  util_error (_("Cannot open mailbox %s: %s"), name,
                              mu_strerror (status));
		  return 1;
		}
	    }

	  status = mu_mailbox_append_message (dest_mbox, msg);
	  if (status)
	    util_error (_("Cannot append message: %s"), mu_strerror (status));
	  else
	    {
	      mu_attribute_set_deleted (attr);
	      saved_count++;
	    }
	}
      else if (mu_attribute_is_deleted (attr))
	/* Skip this one */;
      else if (!keepsave
	       && !mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_PRESERVED)
	       && mu_attribute_is_userflag (attr, MAIL_ATTRIBUTE_SAVED))
	mu_attribute_set_deleted (attr);
      else if (mu_attribute_is_read (attr))
	mu_attribute_set_seen (attr);
    }

  if (saved_count)
    {
      mu_url_t u = NULL;

      mu_mailbox_get_url (dest_mbox, &u);
      fprintf(ofile, 
              ngettext ("Saved %d message in %s\n",
                        "Saved %d messages in %s\n",
			saved_count),
              saved_count, util_url_to_string (u));
      mu_mailbox_close (dest_mbox);
      mu_mailbox_destroy (&dest_mbox);
    }
  return 0;
}
