/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2004, 2005, 2007, 2010 Free Software
   Foundation, Inc.

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
#  include <config.h>
#endif
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <envelope0.h>

int
mu_envelope_create (mu_envelope_t *penvelope, void *owner)
{
  mu_envelope_t envelope;
  if (penvelope == NULL)
    return MU_ERR_OUT_PTR_NULL;
  envelope = calloc (1, sizeof (*envelope));
  if (envelope == NULL)
    return ENOMEM;
  envelope->owner = owner;
  *penvelope = envelope;
  return 0;
}

void
mu_envelope_destroy (mu_envelope_t *penvelope, void *owner)
{
  if (penvelope && *penvelope)
    {
      mu_envelope_t envelope = *penvelope;
      if (envelope->owner == owner)
	{
	  if (envelope->_destroy)
	    envelope->_destroy (envelope);
	  free (envelope->date);
	  free (envelope->sender);
	  free (envelope);
	}
      *penvelope = NULL;
    }
}

void *
mu_envelope_get_owner (mu_envelope_t envelope)
{
  return (envelope) ? envelope->owner : NULL;
}

int
mu_envelope_set_sender (mu_envelope_t envelope,
			int (*_sender) (mu_envelope_t, char *, size_t,
					size_t*),
			void *owner)
{
  if (envelope == NULL)
    return EINVAL;
  if (envelope->owner != owner)
    return EACCES;
  envelope->_get_sender = _sender;
  return 0;
}

int
mu_envelope_set_date (mu_envelope_t envelope,
		      int (*_date) (mu_envelope_t, char *, size_t , size_t *),
		      void *owner)
{
  if (envelope == NULL)
    return EINVAL;
  if (envelope->owner != owner)
    return EACCES;
  envelope->_get_date = _date;
  return 0;
}


/* General accessors: */
#define AC2(a,b) a ## b
#define AC4(a,b,c,d) a ## b ## c ## d
#define ACCESSOR(action,field) AC4(mu_envelope_,action,_,field)

#define DECL_SGET(field)						  \
int									  \
ACCESSOR(sget,field) (mu_envelope_t env, char const **sptr)               \
{									  \
  if (env == NULL)							  \
    return EINVAL;							  \
  if (!env->field)							  \
    {                                                                     \
      if (env->AC2(_get_,field))                                          \
	{								  \
	  size_t n;							  \
	  char *buf;							  \
          int status;                                                     \
	  								  \
	  status = env->AC2(_get_,field) (env, NULL, 0, &n);	          \
	  if (status)							  \
	    return status;						  \
	  								  \
	  buf = malloc (n + 1);						  \
	  if (!buf)							  \
	    return ENOMEM;						  \
	  								  \
	  status = env->AC2(_get_,field) (env, buf, n + 1, NULL);	  \
	  if (status)		                     		          \
            return status;						  \
	  								  \
          env->field = buf;                                               \
	}								  \
      else								  \
        return MU_ERR_NOENT; 	                                          \
    }									  \
  *sptr = env->field;							  \
  return 0;								  \
}

#define DECL_GET(field)							  \
int									  \
ACCESSOR(get,field) (mu_envelope_t env, char *buf, size_t len, size_t *n) \
{									  \
  size_t i;								  \
  const char *str;							  \
  int status = ACCESSOR(sget, field) (env, &str);			  \
  									  \
  if (status)								  \
    return status;							  \
									  \
  i = mu_cpystr (buf, str, len);					  \
  if (n)								  \
    *n = i;								  \
  return 0;								  \
}

#define DECL_AGET(field)						  \
int									  \
ACCESSOR(aget, field) (mu_envelope_t env, char **buf)	                  \
{									  \
  const char *str;							  \
  int status = ACCESSOR(sget, field) (env, &str);			  \
									  \
  if (status)								  \
    return status;							  \
									  \
  if (str)								  \
    {									  \
      *buf = strdup (str);						  \
      if (!*buf)							  \
	status = ENOMEM;						  \
    }									  \
  else									  \
    *buf = NULL;							  \
  return status;							  \
}

#define DECL_ACCESSORS(field)			                          \
DECL_SGET(field)				                          \
DECL_GET(field)					                          \
DECL_AGET(field)

DECL_ACCESSORS(sender)
DECL_ACCESSORS(date)     

