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

#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>
#include <attribute0.h>

int
mu_attribute_create (mu_attribute_t *pattr, void *owner)
{
  mu_attribute_t attr;
  if (pattr == NULL)
    return MU_ERR_OUT_PTR_NULL;
  attr = calloc (1, sizeof(*attr));
  if (attr == NULL)
    return ENOMEM;
  attr->owner = owner;
  *pattr = attr;
  return 0;
}

void
mu_attribute_destroy (mu_attribute_t *pattr, void *owner)
{
  if (pattr && *pattr)
    {
      mu_attribute_t attr = *pattr;
      if (attr->owner == owner)
	free (*pattr);
      /* Loose the link */
      *pattr = NULL;
    }
}

void *
mu_attribute_get_owner (mu_attribute_t attr)
{
  return (attr) ? attr->owner : NULL;
}

int
mu_attribute_is_modified (mu_attribute_t attr)
{
  return (attr) ? attr->flags & MU_ATTRIBUTE_MODIFIED : 0;
}

int
mu_attribute_clear_modified (mu_attribute_t attr)
{
  if (attr)
    attr->flags &= ~MU_ATTRIBUTE_MODIFIED;
  return 0;
}

int
mu_attribute_set_modified (mu_attribute_t attr)
{
  if (attr)
    attr->flags |= MU_ATTRIBUTE_MODIFIED;
  return 0;
}

int
mu_attribute_get_flags (mu_attribute_t attr, int *pflags)
{
  if (attr == NULL)
    return EINVAL;
  if (pflags == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (attr->_get_flags)
    return attr->_get_flags (attr, pflags);
  *pflags = attr->flags;
  return 0;
}

int
mu_attribute_set_flags (mu_attribute_t attr, int flags)
{
  int status = 0;
  int oflags = 0;
  
  if (attr == NULL)
    return EINVAL;

  /* If the required bits are already set, do not modify anything */
  mu_attribute_get_flags (attr, &oflags);
  if ((oflags & flags) == flags)
    return 0;
  
  if (attr->_set_flags)
    status = attr->_set_flags (attr, flags);
  else
    attr->flags |= flags;
  if (status == 0)
    mu_attribute_set_modified (attr);
  return 0;
}

int
mu_attribute_unset_flags (mu_attribute_t attr, int flags)
{
  int status = 0;
  int oflags = 0;

  if (attr == NULL)
    return EINVAL;

  /* If the required bits are already cleared, do not modify anything */
  mu_attribute_get_flags (attr, &oflags);
  if ((oflags & flags) == 0)
    return 0;

  if (attr->_unset_flags)
    status = attr->_unset_flags (attr, flags);
  else
    attr->flags &= ~flags;
  if (status == 0)
    mu_attribute_set_modified (attr);
  return 0;
}

int
mu_attribute_set_get_flags (mu_attribute_t attr, int (*_get_flags)
			 (mu_attribute_t, int *), void *owner)
{
  if (attr == NULL)
    return EINVAL;
  if (attr->owner != owner)
    return EACCES;
  attr->_get_flags = _get_flags;
  return 0;
}

int
mu_attribute_set_set_flags (mu_attribute_t attr, int (*_set_flags)
			 (mu_attribute_t, int), void *owner)
{
  if (attr == NULL)
    return EINVAL;
  if (attr->owner != owner)
    return EACCES;
  attr->_set_flags = _set_flags;
  return 0;
}

int
mu_attribute_set_unset_flags (mu_attribute_t attr, int (*_unset_flags)
			 (mu_attribute_t, int), void *owner)
{
  if (attr == NULL)
    return EINVAL;
  if (attr->owner != owner)
    return EACCES;
  attr->_unset_flags = _unset_flags;
  return 0;
}

/* We add support for "USER" flag, it is a way for external objects
   Not being the owner to add custom flags.  */
int
mu_attribute_set_userflag (mu_attribute_t attr, int flag)
{
  if (attr == NULL)
    return EINVAL;
  attr->user_flags |= flag;
  return 0;
}

int
mu_attribute_set_seen (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_SEEN);
}

int
mu_attribute_set_answered (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_ANSWERED);
}

int
mu_attribute_set_flagged (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_FLAGGED);
}

int
mu_attribute_set_read (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_READ);
}

int
mu_attribute_set_deleted (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_DELETED);
}

int
mu_attribute_set_draft (mu_attribute_t attr)
{
  return mu_attribute_set_flags (attr, MU_ATTRIBUTE_DRAFT);
}

int
mu_attribute_set_recent (mu_attribute_t attr)
{
  int status = mu_attribute_unset_flags (attr, MU_ATTRIBUTE_READ);
  if (status == 0)
    status = mu_attribute_unset_flags (attr, MU_ATTRIBUTE_SEEN);
  return status;
}

int
mu_attribute_is_userflag (mu_attribute_t attr, int flag)
{
  if (attr == NULL)
    return 0;
  return attr->user_flags & flag;
}

int
mu_attribute_is_seen (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_SEEN;
  return 0;
}

int
mu_attribute_is_answered (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_ANSWERED;
  return 0;
}

int
mu_attribute_is_flagged (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_FLAGGED;
  return 0;
}

int
mu_attribute_is_read (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_READ;
  return 0;
}

int
mu_attribute_is_deleted (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_DELETED;
  return 0;
}

int
mu_attribute_is_draft (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return flags & MU_ATTRIBUTE_DRAFT;
  return 0;
}

int
mu_attribute_is_recent (mu_attribute_t attr)
{
  int flags = 0;
  if (mu_attribute_get_flags (attr, &flags) == 0)
    return MU_ATTRIBUTE_IS_UNSEEN(flags);
  return 0;
}

int
mu_attribute_unset_userflag (mu_attribute_t attr, int flag)
{
  if (attr == NULL)
    return 0;
  attr->user_flags &= ~flag;
  return 0;
}

int
mu_attribute_unset_seen (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_SEEN);
}

int
mu_attribute_unset_answered (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_ANSWERED);
}

int
mu_attribute_unset_flagged (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_FLAGGED);
}

int
mu_attribute_unset_read (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_READ);
}

int
mu_attribute_unset_deleted (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_DELETED);
}

int
mu_attribute_unset_draft (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_DRAFT);
}

int
mu_attribute_unset_recent (mu_attribute_t attr)
{
  return mu_attribute_unset_flags (attr, MU_ATTRIBUTE_SEEN);
}

int
mu_attribute_is_equal (mu_attribute_t attr, mu_attribute_t attr2)
{
  int flags2 = 0, flags = 0;
  mu_attribute_get_flags (attr, &flags);
  mu_attribute_get_flags (attr2, &flags2);
  return flags == flags;
}

/*   Miscellaneous.  */
int
mu_attribute_copy (mu_attribute_t dest, mu_attribute_t src)
{
  if (dest == NULL || src == NULL)
    return EINVAL;
  /* Can not be a deep copy.  */
  /* memcpy (dest, src, sizeof (*dest)); */
  dest->flags = src->flags;
  return 0;
}

struct flagtrans
{
  int flag;
  char letter;
};

/* The two macros below are taken from gnulib module verify.h */
#define mu_verify_true(R) \
  (!!sizeof \
   (struct { unsigned int verify_error_if_negative_size__: (R) ? 1 : -1; }))
#define mu_verify(R) extern int (* verify_function__ (void)) [mu_verify_true (R)]

static struct flagtrans flagtrans[] = {
  { MU_ATTRIBUTE_SEEN, 'O' },
  { MU_ATTRIBUTE_ANSWERED, 'A' },
  { MU_ATTRIBUTE_FLAGGED, 'F' },
  { MU_ATTRIBUTE_READ, 'R' },
  { MU_ATTRIBUTE_DELETED, 'd' },
  { 0 }
};

/* If cc reports an error in this statement, fix the MU_STATUS_BUF_SIZE
   declaration in include/mailutils/attribute.h */
mu_verify (MU_ARRAY_SIZE (flagtrans) == MU_STATUS_BUF_SIZE);

int
mu_string_to_flags (const char *buffer, int *pflags)
{
  const char *sep;

  if (pflags == NULL)
    return EINVAL;

  /* Set the attribute */
  if (mu_c_strncasecmp (buffer, "Status:", 7) == 0)
    {
      sep = strchr(buffer, ':'); /* pass the ':' */
      sep++;
    }
  else
    sep = buffer;

  for (; *sep; sep++)
    {
      struct flagtrans *ft;

      for (ft = flagtrans; ft->flag; ft++)
	if (ft->letter == *sep)
	  {
	    *pflags |= ft->flag;
	    break;
	  }
    }
  return 0;
}

/* NOTE: When adding/removing flags, make sure to update the
   MU_STATUS_BUF_SIZE define in include/mailutils/attribute.h */
int
mu_attribute_to_string (mu_attribute_t attr, char *buffer, size_t len,
			size_t *pn)
{
  int flags = 0;
  char buf[MU_STATUS_BUF_SIZE];
  int i;
  int rc;
  struct flagtrans *ft;
  
  rc = mu_attribute_get_flags (attr, &flags);
  if (rc)
    return rc;

  i = 0;
  for (ft = flagtrans; ft->flag; ft++)
    if (ft->flag & flags)
      buf[i++] = ft->letter;
  buf[i++] = 0;

  i = mu_cpystr (buffer, buf, i);
  if (pn)
    *pn = i;
  return 0;
}

