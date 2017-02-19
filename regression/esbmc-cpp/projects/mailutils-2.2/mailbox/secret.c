/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include <string.h>
#include <mailutils/types.h>
#include <mailutils/secret.h>
#include <mailutils/errno.h>

struct _mu_secret
{
  unsigned int refcnt;      /* Number of references to this object */
  size_t length;            /* Secret length */
  unsigned char *obptr;     /* Obfuscated data */
  unsigned char *clptr;     /* Cleartext data */
  unsigned int clref;       /* Number of references to clptr returned
			       this far */
};

static unsigned char xchar;

static void
obfuscate (const unsigned char *input, unsigned char *output, size_t len)
{
  if (!xchar)
    xchar = random () % 255;
  while (len--)
    *output++ = *input++ ^ xchar;
}
  
int
mu_secret_create (mu_secret_t *psec, const char *value, size_t len)
{
  mu_secret_t sec;
  sec = calloc (1, sizeof (sec[0]) + 2 * (len + 1));
  if (!sec)
    return ENOMEM;
  sec->obptr = (unsigned char*)(sec + 1);
  sec->clptr = sec->obptr + len + 1;
  obfuscate ((unsigned char *) value, sec->obptr, len);
  sec->length = len;
  *psec = sec;
  mu_secret_ref (sec);
  return 0;
}

int
mu_secret_dup (mu_secret_t sec, mu_secret_t *newsec)
{
  const char *pass = mu_secret_password (sec);
  int rc = mu_secret_create (newsec, pass, strlen (pass));
  mu_secret_password_unref (sec);
  return rc;
}

void
mu_secret_ref (mu_secret_t sec)
{
  if (sec)
    sec->refcnt++;
}

/* Decrement reference counter in SEC. If it falls to 0, free memory
   allocated for SEC and return 0. Otherwise, return MU_ERR_EXISTS,
   indicating that someone else is still holding it.
   Return EINVAL if sec==NULL. */
int
mu_secret_unref (mu_secret_t sec)
{
  if (sec)
    {
      if (sec->refcnt)
	sec->refcnt--;
      if (sec->refcnt == 0)
	{
	  memset (sec->clptr, 0, sec->length);
	  memset (sec->obptr, 0, sec->length);
	  free (sec);
	  return 0;
	}
      return MU_ERR_EXISTS;
    }
  return EINVAL;
}

void
mu_secret_destroy (mu_secret_t *psec)
{
  if (psec && *psec && mu_secret_unref (*psec) == 0)
    *psec = NULL;
}

const char *
mu_secret_password (mu_secret_t sec)
{
  if (!sec)
    return 0;
  if (sec->clref++ == 0)
    obfuscate (sec->obptr, sec->clptr, sec->length);
  return (const char*) sec->clptr;
}

size_t
mu_secret_length (mu_secret_t sec)
{
  if (!sec)
    return 0;
  return sec->length;
}

void
mu_secret_password_unref (mu_secret_t sec)
{
  if (--sec->clref == 0)
    memset (sec->clptr, 0, sec->length);
}

