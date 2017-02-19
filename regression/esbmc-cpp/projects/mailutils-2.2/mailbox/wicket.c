/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2003, 2004, 2005, 2007, 2009, 2010
   Free Software Foundation, Inc.

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

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>

#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/mu_auth.h>
#include <mailutils/stream.h>

#include <auth0.h>
#include <url0.h>

int
mu_wicket_create (mu_wicket_t *pwicket)
{
  mu_wicket_t wicket = calloc (1, sizeof (*wicket));
  if (!wicket)
    return ENOMEM;
  wicket->refcnt = 1;
  *pwicket = wicket;
  return 0;
}

int
mu_wicket_get_ticket (mu_wicket_t wicket, const char *user, mu_ticket_t *pticket)
{
  if (!wicket)
    return EINVAL;
  if (!pticket)
    return EINVAL;
  if (!wicket->_get_ticket)
    return ENOSYS;
  return wicket->_get_ticket (wicket, wicket->data, user, pticket);
}

int
mu_wicket_ref (mu_wicket_t wicket)
{
  if (!wicket)
    return EINVAL;
  wicket->refcnt++;
  return 0;
}

int
mu_wicket_unref (mu_wicket_t wicket)
{
  if (!wicket)
    return EINVAL;
  if (wicket->refcnt)
    wicket->refcnt--;
  if (wicket->refcnt == 0)
    {
      if (wicket->_destroy)
	wicket->_destroy (wicket);
      free (wicket);
      return 0;
    }
  return MU_ERR_EXISTS;
}


void
mu_wicket_destroy (mu_wicket_t *pwicket)
{
  if (pwicket && *pwicket && mu_wicket_unref (*pwicket) == 0)
    *pwicket = NULL;
}

int
mu_wicket_set_destroy (mu_wicket_t wicket, void (*_destroy) (mu_wicket_t))
{
  if (!wicket)
    return EINVAL;
  wicket->_destroy = _destroy;
  return 0;
}

int
mu_wicket_set_data (mu_wicket_t wicket, void *data)
{
  if (!wicket)
    return EINVAL;
  wicket->data = data;
  return 0;
}

void *
mu_wicket_get_data (mu_wicket_t wicket)
{
  if (!wicket)
    return NULL;
  return wicket->data;
}

int
mu_wicket_set_get_ticket (mu_wicket_t wicket,
			  int (*_get_ticket) (mu_wicket_t, void *,
					      const char *, mu_ticket_t *))
{
  if (!wicket)
    return EINVAL;
  wicket->_get_ticket = _get_ticket;
  return 0;
}


/* A "file wicket" implementation */

struct file_wicket
{
  char *filename;
};

static void
_file_wicket_destroy (mu_wicket_t wicket)
{
  struct file_wicket *fw = mu_wicket_get_data (wicket);
  free (fw->filename);
  free (fw);
}

struct file_ticket
{
  char *filename;
  char *user;
  mu_url_t tickurl;
};

static void
file_ticket_destroy (mu_ticket_t ticket)
{
  struct file_ticket *ft = mu_ticket_get_data (ticket);
  if (ft)
    {
      free (ft->filename);
      free (ft->user);
      mu_url_destroy (&ft->tickurl);
      free (ft);
    }
}

static int get_ticket_url (mu_ticket_t ticket, mu_url_t url, mu_url_t *pticket_url);

int
file_ticket_get_cred (mu_ticket_t ticket, mu_url_t url, const char *challenge,
		      char **pplain, mu_secret_t *psec)
{
  struct file_ticket *ft = mu_ticket_get_data (ticket);

  if (!ft->tickurl)
    {
      int rc = get_ticket_url (ticket, url, &ft->tickurl);
      if (rc)
	return rc;
    }
  if (pplain)
    {
      if (ft->user)
	{
	  *pplain = strdup (ft->user);
	  if (!*pplain)
	    return ENOMEM;
	}
      else
	return mu_url_aget_user (ft->tickurl, pplain);
    }
  else
    return mu_url_get_secret (ft->tickurl, psec);
}

static int
_file_wicket_get_ticket (mu_wicket_t wicket, void *data,
			 const char *user, mu_ticket_t *pticket)
{
  int rc;
  mu_ticket_t ticket;
  struct file_wicket *fw = data;
  struct file_ticket *ft = calloc (1, sizeof (*ft));
  ft->filename = strdup (fw->filename);
  if (!ft->filename)
    {
      free (ft);
      return ENOMEM;
    }
  if (user)
    {
      ft->user = strdup (user);
      if (!ft->user)
	{
	  free (ft->filename);
	  free (ft);
	  return ENOMEM;
	}
    }
  else
    ft->user = NULL;

  rc = mu_ticket_create (&ticket, NULL);
  if (rc)
    {
      free (ft->filename);
      free (ft->user);
      free (ft);
      return rc;
    }
	
  mu_ticket_set_destroy (ticket, file_ticket_destroy, NULL);
  mu_ticket_set_data (ticket, ft, NULL);
  mu_ticket_set_get_cred (ticket, file_ticket_get_cred, NULL);

  *pticket = ticket;
  return 0;
}
  
static int
get_ticket_url (mu_ticket_t ticket, mu_url_t url, mu_url_t *pticket_url)
{
  mu_stream_t stream;
  struct file_ticket *ft = mu_ticket_get_data (ticket);
  int rc;
  mu_url_t u = NULL;
  
  rc = mu_file_stream_create (&stream, ft->filename, MU_STREAM_READ);
  if (rc)
    return rc;
  rc = mu_stream_open (stream);
  if (rc == 0)
    {
      char *buf = NULL;
      size_t bufsize = 0;
      size_t len;

      while ((rc = mu_stream_sequential_getline (stream,
						 &buf, &bufsize, &len)) == 0
	     && len > 0)
	{
	  char *p;
	  int err;
	  
	  /* Truncate a trailing newline. */
	  if (len && buf[len - 1] == '\n')
	    buf[--len] = 0;

	  /* Skip leading spaces  */
	  for (p = buf; *p == ' ' || *p == '\t'; p++)
	    ;
	  /* Skip trailing spaces */
	  for (; len > 0 && (p[len-1] == ' ' || p[len-1] == '\t'); )
	    p[--len] = 0;
	  
	  /* Skip empty lines and comments. */
	  if (*p == 0 || *p == '#')
	    continue;

	  if ((err = mu_url_create (&u, p)) != 0)
	    {
	      /* Skip erroneous entry */
	      /* FIXME: Error message */
	      continue;
	    }
	  if ((err = mu_url_parse (u)) != 0)
	    {
	      /* FIXME: See above */
	      mu_url_destroy (&u);
	      continue;
	    }
	  
	  if (!mu_url_is_ticket (u, url))
	    {
	      mu_url_destroy (&u);
	      continue;
	    }
	  
	  if (ft->user)
	    {
	      if (u->name && strcmp (u->name, "*") != 0
		  && strcmp (ft->user, u->name) != 0)
		{
		  mu_url_destroy (&u);
		  continue;
		}
	    }
	  
	  break;
	}
      mu_stream_close (stream);
      free (buf);
    }
  mu_stream_destroy (&stream, NULL);

  if (rc == 0)
    {
      if (u)
	*pticket_url = u;
      else
	rc = MU_ERR_NOENT;
    }
  
  return rc;
}

int
mu_file_wicket_create (mu_wicket_t *pwicket, const char *filename)
{
  mu_wicket_t wicket;
  int rc;
  struct file_wicket *fw = calloc (1, sizeof (*fw));

  if (!fw)
    return ENOMEM;
  fw->filename = strdup (filename);
  if (!fw->filename)
    {
      free (fw);
      return ENOMEM;
    }
  
  rc = mu_wicket_create (&wicket);
  if (rc)
    {
      free (fw->filename);
      free (fw);
      return rc;
    }
  mu_wicket_set_data (wicket, fw);
  mu_wicket_set_destroy (wicket, _file_wicket_destroy);
  mu_wicket_set_get_ticket (wicket, _file_wicket_get_ticket);
  *pwicket = wicket;
  return 0;
}

