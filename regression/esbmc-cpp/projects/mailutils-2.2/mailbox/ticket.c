/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2007, 2009, 2010 Free
   Software Foundation, Inc.

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
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <termios.h>

#include <mailutils/mutil.h>
#include <mailutils/errno.h>
#include <mailutils/secret.h>
#include <auth0.h>

static void
echo_off (struct termios *stored_settings)
{
  struct termios new_settings;
  tcgetattr (0, stored_settings);
  new_settings = *stored_settings;
  new_settings.c_lflag &= (~ECHO);
  tcsetattr (0, TCSANOW, &new_settings);
}

static void
echo_on (struct termios *stored_settings)
{
  tcsetattr (0, TCSANOW, stored_settings);
}

int
mu_ticket_create (mu_ticket_t *pticket, void *owner)
{
  mu_ticket_t ticket;
  if (pticket == NULL)
    return MU_ERR_OUT_PTR_NULL;
  ticket = calloc (1, sizeof (*ticket));
  if (ticket == NULL)
    return ENOMEM;
  ticket->owner = owner;
  mu_ticket_ref (ticket);
  *pticket = ticket;
  return 0;
}

int
mu_ticket_ref (mu_ticket_t ticket)
{
  if (!ticket)
    return EINVAL;
  ticket->refcnt++;
  return 0;
}
  
int
mu_ticket_unref (mu_ticket_t ticket)
{
  if (!ticket)
    return EINVAL;
  if (ticket->refcnt)
    ticket->refcnt--;
  if (ticket->refcnt == 0)
    {
      if (ticket->plain)
	free (ticket->plain);
      if (ticket->secret)
	mu_secret_destroy (&ticket->secret);
      if (ticket->_destroy)
	ticket->_destroy (ticket);
      free (ticket);
      return 0;
    }
  return MU_ERR_EXISTS;
}
      
void
mu_ticket_destroy (mu_ticket_t *pticket)
{
  if (pticket && *pticket && mu_ticket_unref (*pticket) == 0)
    *pticket = NULL;
}

int
mu_ticket_set_destroy (mu_ticket_t ticket,
		       void (*_destroy) (mu_ticket_t), void *owner)
{
  if (ticket == NULL)
    return EINVAL;
  if (ticket->owner != owner)
    return EACCES;
  ticket->_destroy = _destroy;
  return 0;
}

void *
mu_ticket_get_owner (mu_ticket_t ticket)
{
  return (ticket) ? ticket->owner : NULL;
}

int
mu_ticket_set_get_cred (mu_ticket_t ticket,
			int  (*_get_cred) (mu_ticket_t, mu_url_t,
					   const char *,
					   char **, mu_secret_t *),
			void *owner)
{
  if (ticket == NULL)
    return EINVAL;
  if (ticket->owner != owner)
    return EACCES;
  ticket->_get_cred = _get_cred;
  return 0;
}

int
mu_ticket_set_secret (mu_ticket_t ticket, mu_secret_t secret)
{
  if (ticket == NULL)
    return EINVAL;
  if (ticket->secret)
    mu_secret_unref (ticket->secret);
  mu_secret_ref (secret);
  ticket->secret = secret;
  return 0;
}

int
mu_ticket_set_plain (mu_ticket_t ticket, const char *text)
{
  if (ticket == NULL)
    return EINVAL;
  if (ticket->plain)
    free (ticket->plain);
  ticket->plain = strdup (text);
  if (!ticket->plain)
    return ENOMEM;
  return 0;
}

int
mu_ticket_get_cred (mu_ticket_t ticket, mu_url_t url, const char *challenge,
		    char **pplain, mu_secret_t *psec)
{
  int rc = 0;
  char arg[256];
  
  if (ticket == NULL || (pplain && psec))
    return EINVAL;
  if (pplain == NULL && psec == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (ticket->_get_cred)
    {
      int res = ticket->_get_cred (ticket, url, challenge, pplain, psec);
      if (res == 0)
	return res;
    }

  if (pplain && ticket->plain)
    {
      *pplain = strdup (ticket->plain);
      if (!*pplain)
	return ENOMEM;
    }
  
  if (psec && ticket->secret)
    {
      mu_secret_ref (ticket->secret);
      *psec = ticket->secret;
      return 0;
    }
  
  if (isatty (fileno (stdin)))
    {
      struct termios stored_settings;
      int echo = pplain != NULL;

      printf ("%s", challenge);
      fflush (stdout);
      if (!echo)
	echo_off (&stored_settings);
      fgets (arg, sizeof (arg), stdin);
      if (!echo)
	{
	  echo_on (&stored_settings);
	  putchar ('\n');
	  fflush (stdout);
	}
      arg [strlen (arg) - 1] = '\0'; /* nuke the trailing line.  */
    }

  if (pplain)
    {
      *pplain = strdup (arg);
      if (!*pplain)
	return ENOMEM;
    }
  else
    rc = mu_secret_create (psec, arg, strlen (arg));
  return rc;
}

void *
mu_ticket_get_data (mu_ticket_t ticket)
{
  if (!ticket)
    return NULL;
  return ticket->data;
}

int
mu_ticket_set_data (mu_ticket_t ticket, void *data, void *owner)
{
  if (ticket == NULL)
    return EINVAL;
  if (ticket->owner != owner)
    return EACCES;
  ticket->data = data;
  return 0;
}
