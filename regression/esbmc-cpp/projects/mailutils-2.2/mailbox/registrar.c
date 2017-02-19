/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2005, 2006, 2007, 2009, 2010
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
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/iterator.h>
#include <mailutils/list.h>
#include <mailutils/monitor.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <mailutils/error.h>
#include <mailutils/url.h>
#include <mailutils/mutil.h>
#include <registrar0.h>

/* NOTE: We will leak here since the monitor and the registrar will never
   be released. That's ok we can live with this, it's only done once.  */
static mu_list_t registrar_list;
struct mu_monitor registrar_monitor = MU_MONITOR_INITIALIZER;

static mu_record_t mu_default_record;

void
mu_registrar_set_default_record (mu_record_t record)
{
  mu_default_record = record;
}

int
mu_registrar_get_default_record (mu_record_t *prec)
{
  if (mu_default_record)
    {
      if (prec)
	*prec = mu_default_record;
      return 0;
    }
  return MU_ERR_NOENT;
}

int
mu_registrar_set_default_scheme (const char *scheme)
{
  int status;
  mu_record_t rec;
  
  status = mu_registrar_lookup_scheme (scheme, &rec);
  if (status == 0)
    mu_default_record = rec;
  return status;
}

const char *
mu_registrar_get_default_scheme ()
{
  return mu_default_record ? mu_default_record->scheme : NULL;
}

static int
_registrar_get_list (mu_list_t *plist)
{
  int status = 0;

  if (plist == NULL)
    return MU_ERR_OUT_PTR_NULL;
  mu_monitor_wrlock (&registrar_monitor);
  if (registrar_list == NULL)
    status = mu_list_create (&registrar_list);
  *plist = registrar_list;
  mu_monitor_unlock (&registrar_monitor);
  return status;
}

/* Provided for backward compatibility */
int
mu_registrar_get_list (mu_list_t *plist)
{
  static int warned;

  if (!warned)
    {
      mu_error (_("program uses mu_registrar_get_list(), which is deprecated"));
      warned = 1;
    }
  return _registrar_get_list (plist);
}

int
mu_registrar_get_iterator (mu_iterator_t *pitr)
{
  int status = 0;
  if (pitr == NULL)
    return MU_ERR_OUT_PTR_NULL;
  mu_monitor_wrlock (&registrar_monitor);
  if (registrar_list == NULL)
    {
      status = mu_list_create (&registrar_list);
      if (status)
	return status;
    }
  status = mu_list_get_iterator (registrar_list, pitr);
  mu_monitor_unlock (&registrar_monitor);
  return status;
}

int
mu_registrar_lookup_scheme (const char *scheme, mu_record_t *precord)
{
  size_t len;
  mu_iterator_t iterator;
  int status = mu_registrar_get_iterator (&iterator);
  if (status != 0)
    return status;
  status = MU_ERR_NOENT;
  len = strcspn (scheme, ":");
  for (mu_iterator_first (iterator); !mu_iterator_is_done (iterator);
       mu_iterator_next (iterator))
    {
      mu_record_t record;
      mu_iterator_current (iterator, (void **)&record);
      if (strlen (record->scheme) == len
	  && memcmp (record->scheme, scheme, len) == 0)
	{
	  if (precord)
	    *precord = record;
	  status = 0;
	  break;
	}
    }
  mu_iterator_destroy (&iterator);
  return status;
}

int
mu_registrar_lookup_url (mu_url_t url, int flags,
			 mu_record_t *precord, int *pflags)
{
  mu_iterator_t iterator;
  mu_record_t last_record = NULL;
  int last_flags = 0;
  int status = mu_registrar_get_iterator (&iterator);
  if (status != 0)
    return status;
  status = MU_ERR_NOENT;
  for (mu_iterator_first (iterator); !mu_iterator_is_done (iterator);
       mu_iterator_next (iterator))
    {
      int rc;
      mu_record_t record;
      mu_iterator_current (iterator, (void **)&record);
      if ((rc = mu_record_is_scheme (record, url, flags)))
	{
	  if (rc == flags)
	    {
	      status = 0;
	      last_record = record;
	      last_flags = rc;
	      break;
	    }
	  else if (rc > last_flags)
	    {
	      status = 0;
	      last_record = record;
	      last_flags = rc;
	    }
	}
    }
  mu_iterator_destroy (&iterator);

  if (status == 0)
    {
      if (precord)
	*precord = last_record;
      if (pflags)
	*pflags = last_flags;
    }
  else if (!mu_is_proto (mu_url_to_string (url)) /* FIXME: This check is not
						    enough. */
	   && mu_registrar_get_default_record (precord) == 0)
    {
      status = 0;
      if (pflags)
	*pflags = flags & MU_FOLDER_ATTRIBUTE_FILE; /* FIXME? */
    }
  
  return status;
}

int
mu_registrar_lookup (const char *name, int flags,
		     mu_record_t *precord, int *pflags)
{
  int rc;
  mu_url_t url;
  
  rc = mu_url_create (&url, name);
  if (rc)
    return rc;
  rc = mu_url_parse (url);
  if (rc == 0)
    rc = mu_registrar_lookup_url (url, flags, precord, pflags);
  mu_url_destroy (&url);
  return rc;
}

static int
_compare_prio (const void *item, const void *value)
{
  const mu_record_t a = (const mu_record_t) item;
  const mu_record_t b = (const mu_record_t) value;
  if (a->priority > b->priority)
    return 0;
  return -1;
}

int
mu_registrar_record (mu_record_t record)
{
  int status;
  mu_list_t list;
  mu_list_comparator_t comp;
  
  if (!record)
    return 0;
  _registrar_get_list (&list);
  comp = mu_list_set_comparator (list, _compare_prio);
  status = mu_list_insert (list, record, record, 1);
  if (status == MU_ERR_NOENT)
    status = mu_list_append (list, record);
  mu_list_set_comparator (list, comp);
  return status;
}

int
mu_unregistrar_record (mu_record_t record)
{
  mu_list_t list;
  _registrar_get_list (&list);
  mu_list_remove (list, record);
  return 0;
}

int
mu_record_is_scheme (mu_record_t record, mu_url_t url, int flags)
{
  if (record == NULL)
    return 0;

  /* Overload.  */
  if (record->_is_scheme)
    return record->_is_scheme (record, url, flags);

  if (mu_url_is_scheme (url, record->scheme))
    return MU_FOLDER_ATTRIBUTE_ALL;

  return 0;
}

int
mu_record_set_scheme (mu_record_t record, const char *scheme)
{
  if (record == NULL)
    return EINVAL;
  record->scheme = scheme;
  return 0;
}

int
mu_record_set_is_scheme (mu_record_t record,
			 int (*_is_scheme) (mu_record_t, mu_url_t, int))
{
  if (record == NULL)
    return EINVAL;
  record->_is_scheme = _is_scheme;
  return 0;
}

int
mu_record_get_url (mu_record_t record, int (*(*_purl)) (mu_url_t))
{
  if (record == NULL)
    return EINVAL;
  if (_purl == NULL)
    return MU_ERR_OUT_PTR_NULL;
  /* Overload.  */
  if (record->_get_url)
    return record->_get_url (record, _purl);
  *_purl = record->_url;
  return 0;
}

int
mu_record_set_url (mu_record_t record, int (*_mu_url) (mu_url_t))
{
  if (record == NULL)
    return EINVAL;
  record->_url = _mu_url;
  return 0;
}

int
mu_record_set_get_url (mu_record_t record, int (*_get_url)
		    (mu_record_t, int (*(*)) (mu_url_t)))
{
  if (record == NULL)
    return EINVAL;
  record->_get_url = _get_url;
  return 0;
}

int
mu_record_get_mailbox (mu_record_t record, int (*(*_pmailbox)) (mu_mailbox_t))
{
  if (record == NULL)
    return EINVAL;
  if (_pmailbox == NULL)
    return MU_ERR_OUT_PTR_NULL;
  /* Overload.  */
  if (record->_get_mailbox)
    return record->_get_mailbox (record, _pmailbox);
  *_pmailbox = record->_mailbox;
  return 0;
}

int
mu_record_set_mailbox (mu_record_t record, int (*_mu_mailbox) (mu_mailbox_t))
{
  if (record)
    return EINVAL;
  record->_mailbox = _mu_mailbox;
  return 0;
}

int
mu_record_set_get_mailbox (mu_record_t record, 
     int (*_get_mailbox) (mu_record_t, int (*(*)) (mu_mailbox_t)))
{
  if (record)
    return EINVAL;
  record->_get_mailbox = _get_mailbox;
  return 0;
}

int
mu_record_get_mailer (mu_record_t record, int (*(*_pmailer)) (mu_mailer_t))
{
  if (record == NULL)
    return EINVAL;
  if (_pmailer == NULL)
    return MU_ERR_OUT_PTR_NULL;
  /* Overload.  */
  if (record->_get_mailer)
    return record->_get_mailer (record, _pmailer);
  *_pmailer = record->_mailer;
  return 0;
}

int
mu_record_set_mailer (mu_record_t record, int (*_mu_mailer) (mu_mailer_t))
{
  if (record)
    return EINVAL;
  record->_mailer = _mu_mailer;
  return 0;
}

int
mu_record_set_get_mailer (mu_record_t record, 
  int (*_get_mailer) (mu_record_t, int (*(*)) (mu_mailer_t)))
{
  if (record == NULL)
    return EINVAL;
  record->_get_mailer = _get_mailer;
  return 0;
}

int
mu_record_get_folder (mu_record_t record, int (*(*_pfolder)) (mu_folder_t))
{
  if (record == NULL)
    return EINVAL;
  if (_pfolder == NULL)
    return MU_ERR_OUT_PTR_NULL;
  /* Overload.  */
  if (record->_get_folder)
    return record->_get_folder (record, _pfolder);
  *_pfolder = record->_folder;
  return 0;
}

int
mu_record_set_folder (mu_record_t record, int (*_mu_folder) (mu_folder_t))
{
  if (record == NULL)
    return EINVAL;
  record->_folder = _mu_folder;
  return 0;
}

int
mu_record_set_get_folder (mu_record_t record, 
   int (*_get_folder) (mu_record_t, int (*(*)) (mu_folder_t)))
{
  if (record == NULL)
    return EINVAL;
  record->_get_folder = _get_folder;
  return 0;
}

int
mu_record_list_p (mu_record_t record, const char *name, int flags)
{
  if (record == NULL)
    return EINVAL;
  return record == NULL
          || !record->_list_p
          || record->_list_p (record, name, flags);
}
