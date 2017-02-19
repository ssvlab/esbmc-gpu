/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2005, 2006, 2007, 2009, 2010 Free
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

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>

#include <sys/types.h>

#include <mailutils/errno.h>
#include <mailutils/mutil.h>
#include <mailutils/parse822.h>
#include <mailutils/address.h>
#include <mailutils/cstr.h>

/* Get email addresses from rfc822 address.  */
int
mu_address_create_hint (mu_address_t *a, const char *s, mu_address_t hint,
			int hflags)
{
  /* 'a' must exist, and can't already have been initialized
   */
  int status;

  if (!a)
    return MU_ERR_OUT_PTR_NULL;

  if (!s)
    return EINVAL;

  *a = NULL;
  status = mu_parse822_address_list (a, s, hint, hflags);
  if (status == 0)
    {
      /* And address-list may contain 0 addresses but parse correctly.
       */
      if (!*a)
	return MU_ERR_EMPTY_ADDRESS;

      (*a)->addr = strdup (s);
      if (!(*a)->addr)
	{
	  mu_address_destroy (a);
	  return ENOMEM;
	}
    }
  return status;
}

int
mu_address_create (mu_address_t *a, const char *s)
{
  struct mu_address hint;
  const char *d;
  mu_get_user_email_domain (&d);
  hint.domain = (char*) d;
  return mu_address_create_hint (a, s, &hint, MU_ADDR_HINT_DOMAIN);
}

/* Get email addresses from array of rfc822 addresses.
   FIXME: No hints? */
int
mu_address_createv (mu_address_t *a, const char *sv[], size_t len)
{
  int status = 0;
  size_t buflen = 0;
  char *buf = 0;
  size_t i;

  if (!a)
    return MU_ERR_OUT_PTR_NULL;

  if (!sv)
    return EINVAL;

  if (len == (size_t) - 1)
    {
      const char **vp = sv;

      len = 0;

      for (len = 0; *vp; vp++, len++)
	;
    }

  if (len == 0)
    return EINVAL;

  for (i = 0; i < len; i++)
    {
      /* NULL strings are allowed */
      if (sv[i])
	buflen += strlen (sv[i]);
    }

  buflen += (len - 1) * strlen (", ");
  buflen += 1;			/* Termination null.  */

  buf = malloc (buflen);

  if (!buf)
    return ENOMEM;

  for (i = 0, buf[0] = '\0'; i < len; i++)
    {
      if (i != 0)
	strcat (buf, ", ");

      if (sv[i])
	strcat (buf, sv[i]);
    }

  status = mu_address_create (a, buf);

  free (buf);

  return status;
}

void
mu_address_destroy (mu_address_t *paddress)
{
  if (paddress && *paddress)
    {
      mu_address_t address = *paddress;
      mu_address_t current;
      for (; address; address = current)
	{
	  if (address->addr)
	    free (address->addr);
	  if (address->comments)
	    free (address->comments);
	  if (address->personal)
	    free (address->personal);
	  if (address->email)
	    free (address->email);
	  if (address->local_part)
	    free (address->local_part);
	  if (address->domain)
	    free (address->domain);
	  if (address->route)
	    free (address->route);
	  current = address->next;
	  free (address);
	}
      *paddress = NULL;
    }
}

int
mu_address_concatenate (mu_address_t to, mu_address_t *from)
{
  if (!to || !from || !*from)
    return EINVAL;

  while (to->next)
    to = to->next;

  assert (to && !to->next);

  to->next = *from;
  *from = NULL;

  /* discard the current string cache as it is now inaccurate */
  if (to->addr)
    {
      free (to->addr);
      to->addr = NULL;
    }

  to = to->next;

  /* only the first address must have a cache */
  if (to->addr)
    {
      free (to->addr);
      to->addr = NULL;
    }

  return 0;
}

mu_address_t 
_address_get_nth (mu_address_t addr, size_t no)
{
  int i;
  
  for (i = 1; addr; addr = addr->next, i++)
    if (i == no)
      break;
  return addr;
}

int
mu_address_get_nth (mu_address_t addr, size_t no, mu_address_t *pret)
{
  mu_address_t subaddr = _address_get_nth (addr, no);
  if (!subaddr)
    return MU_ERR_NOENT;
  *pret = mu_address_dup (subaddr);
  return 0;
}


/* General accessors: */
#define AC4(a,b,c,d) a ## b ## c ## d
#define ACCESSOR(action,field) AC4(mu_address_,action,_,field)

#define DECL_SET(field)							\
int									\
ACCESSOR(set, field) (mu_address_t addr, size_t no, const char *buf)	\
{									\
  char *s;								\
  mu_address_t subaddr;							\
  									\
  if (addr == NULL)							\
    return EINVAL;							\
									\
  subaddr = _address_get_nth (addr, no);				\
  if (!subaddr)								\
    return MU_ERR_NOENT;						\
									\
  s = strdup (buf);							\
  if (!s)								\
    return errno;							\
  									\
  free (subaddr->field);						\
  subaddr->field = s;							\
									\
  return 0;								\
}

#define DECL_SGET(field)						\
int									\
ACCESSOR(sget,field) (mu_address_t addr, size_t no, char const **sptr)	\
{									\
  mu_address_t subaddr;							\
  									\
  if (addr == NULL)							\
    return EINVAL;							\
									\
  subaddr = _address_get_nth (addr, no);				\
  if (!subaddr)								\
    return MU_ERR_NOENT;						\
  *sptr = subaddr->field;						\
  return 0;								\
}

#define DECL_GET(field)							  \
int									  \
ACCESSOR(get,field) (mu_address_t addr, size_t no, char *buf, size_t len, \
		     size_t *n)						  \
{									  \
  size_t i;								  \
  const char *str;							  \
  int status = ACCESSOR(sget, field) (addr, no, &str);			  \
  									  \
  if (status)								  \
    return status;							  \
									  \
  i = mu_cpystr (buf, str, len);					  \
  if (n)								  \
    *n = i;								  \
  return 0;								  \
}

#define DECL_AGET(field)						\
int									\
ACCESSOR(aget, field) (mu_address_t addr, size_t no, char **buf)	\
{									\
  const char *str;							\
  int status = ACCESSOR(sget, field) (addr, no, &str);			\
									\
  if (status)								\
    return status;							\
									\
  if (str)								\
    {									\
      *buf = strdup (str);						\
      if (!*buf)							\
	status = ENOMEM;						\
    }									\
  else									\
    *buf = NULL;							\
  return status;							\
}

#define DECL_ACCESSORS(field)			\
DECL_SET(field)					\
DECL_SGET(field)				\
DECL_GET(field)					\
DECL_AGET(field)          



/* Personal part */
DECL_ACCESSORS(personal)
/* Comments */
DECL_ACCESSORS(comments)
/* Email */
DECL_ACCESSORS(email)     
/* Local part */
DECL_ACCESSORS(local_part)
/* Domain */
DECL_ACCESSORS(domain)
/* Route */
DECL_ACCESSORS(route)



#define format_char(c) do {\
 if (buflen) \
   {\
      *buf++ = c;\
      buflen--;\
   }\
 else\
   rc++;\
} while(0) 

#define format_string(str) do {\
 if (buflen) \
   {\
      int n = snprintf (buf, buflen, "%s", str);\
      buf += n;\
      buflen -= n;\
   }\
 else\
   rc += strlen (str);\
} while (0)
     
size_t
mu_address_format_string (mu_address_t addr, char *buf, size_t buflen)
{
  int rc = 0;
  int comma = 0;
  
  for (;addr; addr = addr->next)
    {
      if (addr->email)
	{
	  int space = 0;

	  if (comma)
	    format_char (',');
	  
	  if (addr->personal)
	    {
	      format_char ('"');
	      format_string (addr->personal);
	      format_char ('"');
	      space++;
	    }
	  
	  if (addr->comments)
	    {
	      if (space)
		format_char (' ');
	      format_char ('(');
	      format_string (addr->comments);
	      format_char (')');
	      space++;
	    }
	  
	  if (space)
	    format_char (' ');
	  format_char ('<');
	  format_string (addr->email);
	  format_char ('>');
	  comma++;
	}
    }
  format_char (0);
  return rc;
}

static int
_address_is_group (mu_address_t addr)
{
  if (addr->personal && !addr->local_part && !addr->domain)
    return 1;
  return 0;
}

static int
_address_is_email (mu_address_t addr)
{
  if (addr->email)
    return 1;
  return 0;
}

static int
_address_is_unix_mailbox (mu_address_t addr)
{
  if (addr->local_part && !addr->email)
    return 1;
  return 0;
}

int
mu_address_is_group (mu_address_t addr, size_t no, int *yes)
{
  mu_address_t subaddr;
  
  if (addr == NULL)
    return EINVAL;

  subaddr = _address_get_nth (addr, no);
  if (!subaddr)
    return MU_ERR_NOENT;
  
  if (yes)
    *yes = _address_is_group (subaddr);
  return 0;
}

int
mu_address_to_string (mu_address_t addr, char *buf, size_t len, size_t *n)
{
  size_t i;
  if (addr == NULL)
    return EINVAL;
  if (buf)
    *buf = '\0';

  if (!addr->addr)
    {
      i = mu_address_format_string (addr, NULL, 0);
      addr->addr = malloc (i + 1);
      if (!addr->addr)
	return ENOMEM;
      mu_address_format_string (addr, addr->addr, i+1);
    }
      
  i = mu_cpystr (buf, addr->addr, len);
  if (n)
    *n = i;
  return 0;
}

int
mu_address_get_count (mu_address_t addr, size_t *pcount)
{
  size_t j;
  for (j = 0; addr; addr = addr->next, j++)
    ;
  if (pcount)
    *pcount = j;
  return 0;
}

int
mu_address_get_group_count (mu_address_t addr, size_t *pcount)
{
  size_t j;
  for (j = 0; addr; addr = addr->next)
    {
      if (_address_is_group (addr))
	j++;
    }
  if (pcount)
    *pcount = j;
  return 0;
}

int
mu_address_get_email_count (mu_address_t addr, size_t *pcount)
{
  size_t j;
  for (j = 0; addr; addr = addr->next)
    {
      if (_address_is_email (addr))
	j++;
    }
  if (pcount)
    *pcount = j;
  return 0;
}

int
mu_address_get_unix_mailbox_count (mu_address_t addr, size_t *pcount)
{
  size_t j;
  for (j = 0; addr; addr = addr->next)
    {
      if (_address_is_unix_mailbox (addr))
	j++;
    }
  if (pcount)
    *pcount = j;
  return 0;
}

int
mu_address_contains_email (mu_address_t addr, const char *email)
{
  for (; addr; addr = addr->next)
    if (mu_c_strcasecmp (addr->email, email) == 0)
      return 1;
  return 0;
}

mu_address_t
mu_address_dup (mu_address_t src)
{
  mu_address_t dst = calloc (1, sizeof (*dst));

  if (!dst)
    return NULL;

  /* FIXME: How about:
    if (src->addr)
      dst->addr = strdup (src->addr);
    ?
  */
  if (src->comments)
    dst->comments = strdup (src->comments);
  if (src->personal)
    dst->personal = strdup (src->personal);
  if (src->email)
    dst->email = strdup (src->email);
  if (src->local_part)
    dst->local_part = strdup (src->local_part);
  if (src->domain)
    dst->domain = strdup (src->domain);
  if (src->route)
    dst->route = strdup (src->route);

  return dst;
}
  
int
mu_address_union (mu_address_t *a, mu_address_t b)
{
  mu_address_t last = NULL;
    
  if (!a || !b)
    return EINVAL;

  if (!*a)
    {
      *a = mu_address_dup (b);
      if (!*a)
	return ENOMEM;
      last = *a;
      b = b->next;
    }
  else
    {
      if ((*a)->addr)
	{
	  free ((*a)->addr);
	  (*a)->addr = NULL;
	}
      for (last = *a; last->next; last = last->next)
	;
    }

  for (; b; b = b->next)
    if (!mu_address_contains_email (*a, b->email))
      {
	mu_address_t next = mu_address_dup (b);
	if (!next)
	  return ENOMEM;
	last->next = next;
	last = next;
      }
  return 0;
}
  
