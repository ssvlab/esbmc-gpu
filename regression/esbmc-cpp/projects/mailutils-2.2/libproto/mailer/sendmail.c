/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2008, 2009,
   2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_SENDMAIL

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <confpaths.h>

#include <mailutils/address.h>
#include <mailutils/debug.h>
#include <mailutils/observer.h>
#include <mailutils/property.h>
#include <mailutils/url.h>
#include <mailutils/errno.h>
#include <mailutils/progmailer.h>

#include <url0.h>
#include <mailer0.h>
#include <registrar0.h>

static void sendmail_destroy (mu_mailer_t);
static int sendmail_open (mu_mailer_t, int);
static int sendmail_close (mu_mailer_t);
static int sendmail_send_message (mu_mailer_t, mu_message_t, mu_address_t,
				  mu_address_t);


static int
_url_sendmail_init (mu_url_t url)
{
  /* not valid in a sendmail url */
  if (url->user || url->secret || url->auth || url->qargc
      || url->host || url->port)
    return EINVAL;

  if (url->path == 0)
    if ((url->path = strdup (PATH_SENDMAIL)) == 0)
      return ENOMEM;

  return 0;
}

int
_mu_mailer_sendmail_init (mu_mailer_t mailer)
{
  int status;
  mu_progmailer_t pm;

  status = mu_progmailer_create (&pm);
  if (status)
    return status;
  
  mailer->data = pm;
  mailer->_destroy = sendmail_destroy;
  mailer->_open = sendmail_open;
  mailer->_close = sendmail_close;
  mailer->_send_message = sendmail_send_message;

  /* Set our properties.  */
  {
    mu_property_t property = NULL;
    mu_mailer_get_property (mailer, &property);
    mu_property_set_value (property, "TYPE", "SENDMAIL", 1);
  }
  return 0;
}

static void
sendmail_destroy (mu_mailer_t mailer)
{
  mu_progmailer_destroy ((mu_progmailer_t*)&mailer->data);
}

static int
sendmail_open (mu_mailer_t mailer, int flags)
{
  mu_progmailer_t pm = mailer->data;
  int status;
  const char *path;

  /* Sanity checks.  */
  if (pm == NULL)
    return EINVAL;

  mailer->flags = flags;

  if ((status = mu_url_sget_path (mailer->url, &path)))
    return status;

  if (access (path, X_OK) == -1)
    return errno;
  mu_progmailer_set_debug (pm, mailer->debug);
  status = mu_progmailer_set_command (pm, path);
  MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE, "sendmail (%s)\n", path);
  return status;
}

static int
sendmail_close (mu_mailer_t mailer)
{
  return mu_progmailer_close (mailer->data);
}

static int
mailer_property_is_set (mu_mailer_t mailer, const char *name)
{
  mu_property_t property = NULL;

  mu_mailer_get_property (mailer, &property);
  return mu_property_is_set (property, name);
}

static int
sendmail_send_message (mu_mailer_t mailer, mu_message_t msg, mu_address_t from,
		       mu_address_t to)
{
  mu_progmailer_t pm = mailer->data;
  int argc = 0;
  const char **argvec = NULL;
  size_t tocount = 0;
  const char *emailfrom = NULL;
  int status;
  
  if (!pm)
    return EINVAL;
  
  /* Count the length of the arg vec: */

  argc++;			/* terminating NULL */
  argc++;			/* sendmail */
  argc++;			/* -oi (do not treat '.' as message
				   terminator) */
  
  if (from)
    {
      if ((status = mu_address_sget_email (from, 1, &emailfrom)) != 0)
	{
	  MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		     "cannot get recipient email: %s\n",
		     mu_strerror (status));
	  return status;
	}

      if (!emailfrom)
	{
	  /* the address wasn't fully qualified, choke (for now) */
	  MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE,
		     "envelope from (%s) not fully qualifed\n",
		     emailfrom);
	  return MU_ERR_BAD_822_FORMAT;
	}

      argc += 2;		/* -f from */
    }
	
  if (to)
    {
      status = mu_address_get_email_count (to, &tocount);
      if (status)
	return status;

      if (tocount == 0)
	{
	  MU_DEBUG (mailer->debug, MU_DEBUG_TRACE,
		    "missing recipients\n");
	  return MU_ERR_NOENT;
	}
      
      argc += tocount;	/* 1 per to address */
    }

  argc++;		/* -t */

  /* Allocate arg vec: */
  if ((argvec = calloc (argc, sizeof (*argvec))) == 0)
    return ENOMEM;
  
  argc = 0;
  
  if (mu_progmailer_sget_command (pm, &argvec[argc]) || argvec[argc] == NULL)
    {
      free (argvec);
      return EINVAL;
    }
  
  argc++;
  argvec[argc++] = "-oi";

  if (from)
    {
      argvec[argc++] = "-f";
      argvec[argc++] = emailfrom;
    }
	
  if (!to || mailer_property_is_set (mailer, "READ_RECIPIENTS"))
    {
      argvec[argc++] = "-t";
    }
  else
    {
      size_t i;
      size_t count = 0;
      
      mu_address_get_count (to, &count);
      
      for (i = 1; i <= count; i++)
	{
	  const char *email;
	  if ((status = mu_address_sget_email (to, i, &email)) != 0)
	    {
	      free (argvec);
	      MU_DEBUG2 (mailer->debug, MU_DEBUG_ERROR,
			 "cannot get email of recipient #%lu: %s\n",
			 (unsigned long) i, mu_strerror (status));
	      return status;
	    }
	  
	  if (!email)
	    {
	      MU_DEBUG1 (mailer->debug, MU_DEBUG_TRACE,
			 "envelope to (%s) not fully qualifed\n",
			 email);
	      free (argvec);
	      return MU_ERR_BAD_822_FORMAT;
	    }
	  argvec[argc++] = email;
	}
    }
  argvec[argc] = NULL;
  
  mu_progmailer_set_debug (pm, mailer->debug);
  status = mu_progmailer_open (pm, (char**) argvec);
  if (status == 0)
    {
      status = mu_progmailer_send (pm, msg);
      if (status == 0)
	  mu_observable_notify (mailer->observable, MU_EVT_MAILER_MESSAGE_SENT,
				msg);
      else
	MU_DEBUG1 (mailer->debug, MU_DEBUG_ERROR,
		   "progmailer error: %s\n",
		   mu_strerror (status));
    }
  
  free (argvec);
  return status;
}

static struct _mu_record _sendmail_record =
{
  MU_SENDMAIL_PRIO,
  MU_SENDMAIL_SCHEME,
  _url_sendmail_init,    /* url init.  */
  _mu_mailer_mailbox_init,     /* Mailbox entry.  */
  _mu_mailer_sendmail_init, /* Mailer entry.  */
  _mu_mailer_folder_init, /* Folder entry.  */
  NULL, /* No need for a back pointer.  */
  NULL, /* _is_scheme method.  */
  NULL, /* _get_url method.  */
  NULL, /* _get_mailbox method.  */
  NULL, /* _get_mailer method.  */
  NULL  /* _get_folder method.  */
};
/* We export, url parsing and the initialisation of
   the mailbox, via the register entry/record.  */
mu_record_t mu_sendmail_record = &_sendmail_record;

#else
#include <stdio.h>
#include <registrar0.h>
mu_record_t mu_sendmail_record = NULL;
#endif
