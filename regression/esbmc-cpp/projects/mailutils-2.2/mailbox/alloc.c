/* Error-proof memory allocation functions.
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 3, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mailutils/error.h>
#include <mailutils/nls.h>
#include <mailutils/alloc.h>

void (*mu_alloc_die_hook) (void) = NULL;

void
mu_alloc_die ()
{
  if (mu_alloc_die_hook)
    mu_alloc_die_hook ();
  mu_error (_("Not enough memory"));
  abort ();
}

void *
mu_alloc (size_t size)
{
  void *p = malloc (size);
  if (!p)
    mu_alloc_die ();
  return p;
}

void *
mu_calloc (size_t nmemb, size_t size)
{
  void *p = calloc (nmemb, size);
  if (!p)
    mu_alloc_die ();
  return p;
}  

void *
mu_zalloc (size_t size)
{
  void *p = mu_alloc (size);
  memset (p, 0, size);
  return p;
}

void *
mu_realloc (void *p, size_t size)
{
  void *newp = realloc (p, size);
  if (!newp)
    mu_alloc_die ();
  return newp;
}

char *
mu_strdup (const char *s)
{
  char *news = strdup (s);
  if (!news)
    mu_alloc_die ();
  return news;
}

/* Copied from gnulib */
void *
mu_2nrealloc (void *p, size_t *pn, size_t s)
{
  size_t n = *pn;
  
  if (!p)
    {
      if (!n)
	{
	  /* The approximate size to use for initial small allocation
	     requests, when the invoking code specifies an old size of
	     zero.  64 bytes is the largest "small" request for the
	     GNU C library malloc.  */
	  enum { DEFAULT_MXFAST = 64 };

	  n = DEFAULT_MXFAST / s;
	  n += !n;
	}
    }
  else
    {
      /* Set N = ceil (1.5 * N) so that progress is made if N == 1.
	 Check for overflow, so that N * S stays in size_t range.
	 The check is slightly conservative, but an exact check isn't
	 worth the trouble.  */
      if ((size_t) -1 / 3 * 2 / s <= n)
	mu_alloc_die ();
      n += (n + 1) / 2;
    }

  *pn = n;
  return mu_realloc (p, n * s);
}

