/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

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

#include "maidag.h"

void
make_tmp (const char *from, mu_mailbox_t *mbox)
{
  struct mail_tmp *mtmp;
  char *buf = NULL;
  size_t n = 0;
  int rc;

  if (mail_tmp_begin (&mtmp, from))
    exit (EX_TEMPFAIL);

  while (getline (&buf, &n, stdin) > 0)
    if ((rc = mail_tmp_add_line (mtmp, buf, strlen (buf))))
      break;
  free (buf);
  if (rc == 0)
    rc = mail_tmp_finish (mtmp, mbox);
  mail_tmp_destroy (&mtmp);
  if (rc)
    exit (EX_TEMPFAIL);
}

int
mda (mu_mailbox_t mbx, char *username)
{
  int status;
  mu_message_t msg;
  
  if ((status = mu_mailbox_get_message (mbx, 1, &msg)) != 0)
    {
      maidag_error (_("cannot get message from the temporary mailbox: %s"),
		    mu_strerror (status));
      return EX_TEMPFAIL;
    }

  deliver (msg, username, NULL);

  if (multiple_delivery)
    exit_code = EX_OK;

  return exit_code;
}

int
maidag_stdio_delivery (int argc, char **argv)
{
  mu_mailbox_t mbox;

  make_tmp (sender_address, &mbox);
  
  if (multiple_delivery)
    multiple_delivery = argc > 1;

  for (; *argv; argv++)
    mda (mbox, *argv);
  return exit_code;
}

static int biff_fd = -1;
static struct sockaddr_in biff_in;
static const char *biff_user_name;

static int
notify_action (mu_observer_t obs, size_t type, void *data, void *action_data)
{
  if (type == MU_EVT_MESSAGE_APPEND && biff_user_name)
    {
      mu_message_qid_t qid = data;
      mu_mailbox_t mbox = mu_observer_get_owner (obs);
      mu_url_t url;
      char *buf;
      
      mu_mailbox_get_url (mbox, &url);
      asprintf (&buf, "%s@%s:%s", biff_user_name,
		qid, mu_url_to_string (url));
      if (buf)
	{
	  sendto (biff_fd, buf, strlen (buf), 0,
		  (struct sockaddr *)&biff_in, sizeof biff_in);
	  free (buf);
	}
    }
  return 0;
}

static void
attach_notify (mu_mailbox_t mbox)
{
  struct servent *sp;
  mu_observer_t observer;
  mu_observable_t observable;

  if (biff_fd == -1)
    {
      if ((sp = getservbyname ("biff", "udp")) == NULL)
	{
	  biff_fd = -2;
	  return;
	}
      biff_in.sin_family = AF_INET;
      biff_in.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
      biff_in.sin_port = sp->s_port;
      
      biff_fd = socket (PF_INET, SOCK_DGRAM, 0);
      if (biff_fd < 0)
	{
	  biff_fd = -2;
	  return;
	}
    }

  if (biff_fd)
    {
      mu_observer_create (&observer, mbox);
      mu_observer_set_action (observer, notify_action, mbox);
      mu_mailbox_get_observable (mbox, &observable);
      mu_observable_attach (observable, MU_EVT_MESSAGE_APPEND, observer);
    }
}  

int
deliver_to_user (mu_mailbox_t mbox, mu_message_t msg,
		 struct mu_auth_data *auth,
		 char **errp)
{
  int status;
  char *path;
  mu_url_t url = NULL;
  mu_locker_t lock;
  int failed = 0;
  
  mu_mailbox_get_url (mbox, &url);
  path = (char*) mu_url_to_string (url);

  status = mu_mailbox_open (mbox, MU_STREAM_APPEND|MU_STREAM_CREAT);
  if (status != 0)
    {
      maidag_error (_("cannot open mailbox %s: %s"), 
                    path, mu_strerror (status));
      return EX_TEMPFAIL;
    }

  attach_notify (mbox);
  
  /* FIXME: This is superfluous, as mu_mailbox_append_message takes care
     of locking anyway. But I leave it here for the time being. */
  mu_mailbox_get_locker (mbox, &lock);

  if (lock)
    {
      status = mu_locker_lock (lock);

      if (status)
	{
	  maidag_error (_("cannot lock mailbox `%s': %s"), path,
		        mu_strerror (status));
	  exit_code = EX_TEMPFAIL;
	  return EX_TEMPFAIL;
	}
    }
  
#if defined(USE_MAILBOX_QUOTAS)
  if (auth)
    {
      mu_off_t n;
      size_t msg_size;
      mu_off_t mbsize;
      
      if ((status = mu_mailbox_get_size (mbox, &mbsize)))
	{
	  maidag_error (_("cannot get size of mailbox %s: %s"),
			path, mu_strerror (status));
	  if (status == ENOSYS)
	    mbsize = 0; /* Try to continue anyway */
	  else
	    return EX_TEMPFAIL;
	}
    
      switch (check_quota (auth, mbsize, &n))
	{
	case MQUOTA_EXCEEDED:
	  maidag_error (_("%s: mailbox quota exceeded for this recipient"),
			auth->name);
	  if (errp)
	    asprintf (errp, "%s: mailbox quota exceeded for this recipient",
		      auth->name);
	  exit_code = EX_QUOTA();
	  failed++;
	  break;
	  
	case MQUOTA_UNLIMITED:
	  break;
	  
	default:
	  if ((status = mu_message_size (msg, &msg_size)))
	    {
	      maidag_error (_("cannot get message size (input message %s): %s"),
			    path, mu_strerror (status));
	      exit_code = EX_UNAVAILABLE;
	      failed++;
	    }
	  else if (msg_size > n)
	    {
	      maidag_error (_("%s: message would exceed maximum mailbox size for "
			      "this recipient"),
			    auth->name);
	      if (errp)
		asprintf (errp,
			  "%s: message would exceed maximum mailbox size "
			  "for this recipient",
			  auth->name);
	      exit_code = EX_QUOTA();
	      failed++;
	    }
	  break;
	}
    }
#endif
  
  if (!failed && switch_user_id (auth, 1) == 0)
    {
      status = mu_mailbox_append_message (mbox, msg);
      if (status)
	{
	  maidag_error (_("error writing to mailbox %s: %s"),
		        path, mu_strerror (status));
	  failed++;
	}
      else
	{
	  status = mu_mailbox_sync (mbox);
	  if (status)
	    {
	      maidag_error (_("error flushing mailbox %s: %s"),
			    path, mu_strerror (status));
	      failed++;
	    }
	}
      switch_user_id (auth, 0);
    }

  mu_mailbox_close (mbox);
  mu_locker_unlock (lock);
  return failed ? exit_code : 0;
}

static int
is_remote_url (mu_url_t url)
{
  const char *scheme;
  int rc = mu_url_sget_scheme (url, &scheme);
  return rc == 0 && strncmp (scheme, "remote+", 7) == 0;
}

static int
is_mailer_url (mu_url_t url)
{
  mu_record_t record = NULL;
  int (*pfn) (mu_mailer_t) = NULL;
  
  return mu_registrar_lookup_url (url, MU_FOLDER_ATTRIBUTE_FILE,
				  &record, NULL) == 0
         && mu_record_get_mailer (record, &pfn) == 0
         && pfn;
}

int
deliver_url (mu_url_t url, mu_message_t msg, const char *name, char **errp)
{
  struct mu_auth_data *auth = NULL;
  mu_mailbox_t mbox;
  int status;

  if (name)
    {
      auth = mu_get_auth_by_name (name);
      if (!auth)
	{
	  maidag_error (_("%s: no such user"), name);
	  if (errp)
	    asprintf (errp, "%s: no such user", name);
	  exit_code = EX_NOUSER;
	  return EX_NOUSER;
	}

      if (current_uid)
	auth->change_uid = 0;

      if (switch_user_id (auth, 1))
	return EX_TEMPFAIL;
      status = script_apply (msg, auth);
      if (switch_user_id (auth, 0))
	return EX_TEMPFAIL;
      if (status)
	{
	  exit_code = EX_OK;
	  mu_auth_data_free (auth);
	  return 0;
	}
 
      if (forward_file)
	switch (maidag_forward (msg, auth, forward_file))
	  {
	  case maidag_forward_none:
	  case maidag_forward_metoo:
	    break;
	    
	  case maidag_forward_ok:
	    mu_auth_data_free (auth);
	    return 0;

	  case maidag_forward_error:
	    mu_auth_data_free (auth);
	    return exit_code = EX_TEMPFAIL;
	  }
    }
  
  if (!url)
    {
      status = mu_url_create (&url, auth->mailbox);
      if (status)
	{
	  maidag_error (_("cannot create URL for %s: %s"),
			auth->mailbox, mu_strerror (status));
	  return exit_code = EX_UNAVAILABLE;
	}
      status = mu_url_parse (url);
      if (status)
	{
	  maidag_error (_("error parsing URL %s: %s"),
			auth->mailbox, mu_strerror (status));
	  return exit_code = EX_UNAVAILABLE;
	}
    }      

  status = mu_mailbox_create_from_url (&mbox, url);

  if (status)
    {
      maidag_error (_("cannot open mailbox %s: %s"),
		    mu_url_to_string (url),
		    mu_strerror (status));
      mu_url_destroy (&url);
      mu_auth_data_free (auth);
      return EX_TEMPFAIL;
    }

  biff_user_name = name;
  
  /* Actually open the mailbox. Switch to the user's euid to make
     sure the maildrop file will have right privileges, in case it
     will be created */
  if (switch_user_id (auth, 1))
    return EX_TEMPFAIL;
  status = deliver_to_user (mbox, msg, auth, errp);
  if (switch_user_id (auth, 0))
    return EX_TEMPFAIL;

  mu_auth_data_free (auth);
  mu_mailbox_destroy (&mbox);

  return status;
}

int
deliver (mu_message_t msg, char *dest_id, char **errp)
{
  int status;
  const char *name;
  mu_url_t url = NULL;
  
  if (url_option)
    {
      status = mu_url_create (&url, dest_id);
      if (status)
	{
	  maidag_error (_("%s: cannot create url: %s"), dest_id,
			mu_strerror (status));
	  return EX_NOUSER;
	}
      status = mu_url_parse (url);
      if (status)
	{
	  maidag_error (_("%s: cannot parse url: %s"), dest_id,
			mu_strerror (status));
	  mu_url_destroy (&url);
	  return EX_NOUSER;
	}
      status = mu_url_sget_user (url, &name);
      if (status == MU_ERR_NOENT)
	{
	  if (!is_mailer_url (url) && !is_remote_url (url))
	    {
	      maidag_error (_("no user name"));
	      if (errp)
		asprintf (errp, "no such user");
	      exit_code = EX_NOUSER;
	      return EX_NOUSER;
	    }
	  else
	    name = NULL;
	}
      else if (status)
	{
	  maidag_error (_("%s: cannot get user name from url: %s"),
			dest_id, mu_strerror (status));
	  mu_url_destroy (&url);
	  return EX_NOUSER;
	}
    }
  else
    {
      name = dest_id;
      dest_id = NULL;
    }
  return deliver_url (url, msg, name, errp);
}
  
