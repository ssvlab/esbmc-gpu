/* GNU mailutils - a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Library Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Library General Public License for more details.

   You should have received a copy of the GNU Library General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <string.h>
#include <strings.h>

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <mailutils/sys/nntp.h>
#include <mailutils/iterator.h>

static int  nntp_itr_dup     (void **ptr, void *owner);
static int  nntp_itr_destroy (mu_iterator_t itr, void *owner);
static int  nntp_itr_first   (void *owner);
static int  nntp_itr_next    (void *woner);
static int  nntp_itr_getitem (void *owner, void **pret, const void **pkey);
static int  nntp_itr_curitem_p (void *owner, void *data);
static int  nntp_itr_finished_p (void *owner);

struct nntp_iterator
{
  mu_nntp_t nntp;
  int done;
  char *item;
};

int
mu_nntp_iterator_create (mu_nntp_t nntp, mu_iterator_t *piterator)
{
  struct nntp_iterator *nntp_iterator;
  mu_iterator_t iterator;
  int status;

  nntp_iterator = malloc (sizeof *nntp_iterator);
  if (nntp_iterator == NULL)
    return ENOMEM;

  nntp_iterator->item = NULL;
  nntp_iterator->done = 0;
  nntp_iterator->nntp= nntp;

  status = mu_iterator_create (&iterator, nntp_iterator);
  if (status != 0)
    {
      free (nntp_iterator);
      return status;
    }
  mu_iterator_set_first (iterator, nntp_itr_first);
  mu_iterator_set_next (iterator, nntp_itr_next);
  mu_iterator_set_getitem (iterator, nntp_itr_getitem);
  mu_iterator_set_finished_p (iterator, nntp_itr_finished_p);
  mu_iterator_set_curitem_p (iterator, nntp_itr_curitem_p);
  mu_iterator_set_destroy (iterator, nntp_itr_destroy);
  mu_iterator_set_dup (iterator, nntp_itr_dup);

  *piterator = iterator;
  return 0;
}

static int
nntp_itr_dup (void **ptr, void *owner)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  struct nntp_iterator *clone = malloc (sizeof *nntp_iterator);
  if (clone == NULL)
    return ENOMEM;
  /* let the assignement operator copy the elements.  */
  *clone = *nntp_iterator;
  *ptr = clone;
  return 0;
}

static int
nntp_itr_destroy (mu_iterator_t iterator, void *owner)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  /* Delicate situation if they did not finish to drain the result
     We take te approach to do it for the user.  FIXME: Not sure
     if this is the rigth thing to do. The other way is to close the stream  */
  if (!nntp_iterator->done)
    {
      char buf[128];
      size_t n = 0;
      while (mu_nntp_readline (nntp_iterator->nntp, buf, sizeof buf, &n) > 0 && n > 0)
	n = 0;
    }
  if (nntp_iterator->item)
    free (nntp_iterator->item);
  nntp_iterator->nntp->state = MU_NNTP_NO_STATE;
  free (nntp_iterator);
  return 0;
}

static int
nntp_itr_first  (void *data)
{
  return nntp_itr_next (data);
}

static int
nntp_itr_next (void *owner)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  size_t n = 0;
  int status = 0;

  if (!nntp_iterator->done)
    {
      /* The first readline will not consume the buffer, we just need to
	 know how much to read.  */
      status = mu_nntp_readline (nntp_iterator->nntp, NULL, 0, &n);
      if (status == 0)
	{
	  if (n)
	    {
	      char *buf;
	      buf = calloc (n + 1, 1);
	      if (buf)
		{
		  /* Consume.  */
		  mu_nntp_readline (nntp_iterator->nntp, buf, n + 1, NULL);
		  if (buf[n - 1] == '\n')
		    buf[n - 1] = '\0';
		  if (nntp_iterator->item)
		    free (nntp_iterator->item);
		  nntp_iterator->item = buf;
		}
	      else
		status = ENOMEM;
	    }
	  else
	    {
	      nntp_iterator->done = 1;
	      nntp_iterator->nntp->state = MU_NNTP_NO_STATE;
	    }
	}
    }
  return status;
}

static int
nntp_itr_getitem (void *owner, void **item, const void **pkey)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  if (item)
    {
      *((char **)item) = nntp_iterator->item;
      nntp_iterator->item = NULL;
    }
  if (pkey)
    *pkey = NULL;
  return 0;
}

static int
nntp_itr_finished_p (void *owner)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  return nntp_iterator->done;
}

static int
nntp_itr_curitem_p (void *owner, void *item)
{
  struct nntp_iterator *nntp_iterator = (struct nntp_iterator *)owner;
  return *((char **)item) == nntp_iterator->item;
}
