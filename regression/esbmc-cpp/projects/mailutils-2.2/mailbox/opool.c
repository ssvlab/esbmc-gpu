/* String-list functions for GNU Mailutils.
   Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.

   Based on slist module from GNU Radius.  Written by Sergey Poznyakoff.
   
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
#include <mailutils/types.h>
#include <mailutils/alloc.h>
#include <mailutils/opool.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/nls.h>
#include <mailutils/iterator.h>

struct mu_opool_bucket
{
  struct mu_opool_bucket *next;
  char *buf;
  size_t level;
  size_t size;
};

struct _mu_opool
{
  int memerr;
  size_t bucket_size;
  size_t itr_count;
  struct mu_opool_bucket *head, *tail;
  struct mu_opool_bucket *free;
};

static struct mu_opool_bucket *
alloc_bucket (struct _mu_opool *opool, size_t size)
{
  struct mu_opool_bucket *p = malloc (sizeof (*p) + size);
  if (!p)
    {
      if (opool->memerr)
	mu_alloc_die ();
    }
  else
    {
      p->buf = (char*)(p + 1);
      p->level = 0;
      p->size = size;
      p->next = NULL;
    }
  return p;
}

static int
alloc_pool (mu_opool_t opool, size_t size)
{
  struct mu_opool_bucket *p = alloc_bucket (opool, opool->bucket_size);
  if (!p)
    return ENOMEM;
  if (opool->tail)
    opool->tail->next = p;
  else
    opool->head = p;
  opool->tail = p;
  return 0;
}

static int
copy_chars (mu_opool_t opool, const char *str, size_t n, size_t *psize)
{
  size_t rest;

  if (!opool->head || opool->tail->level == opool->tail->size)
    if (alloc_pool (opool, opool->bucket_size))
      return ENOMEM;
  rest = opool->tail->size - opool->tail->level;
  if (n > rest)
    n = rest;
  memcpy (opool->tail->buf + opool->tail->level, str, n);
  opool->tail->level += n;
  *psize = n;
  return 0;
}

int
mu_opool_create (mu_opool_t *pret, int memerr)
{
  struct _mu_opool *x = malloc (sizeof (x[0]));
  if (!x)
    {
      if (memerr)
	mu_alloc_die ();
      return ENOMEM;
    }
  x->memerr = memerr;
  x->bucket_size = MU_OPOOL_BUCKET_SIZE;
  x->itr_count = 0;
  x->head = x->tail = x->free = 0;
  *pret = x;
  return 0;
}

int
mu_opool_set_bucket_size (mu_opool_t opool, size_t size)
{
  if (!opool)
    return EINVAL;
  opool->bucket_size = size;
  return 0;
}

int
mu_opool_get_bucket_size (mu_opool_t opool, size_t *psize)
{
  if (!opool || !psize)
    return EINVAL;
  *psize = opool->bucket_size;
  return 0;
}

void
mu_opool_clear (mu_opool_t opool)
{
  if (!opool)
    return;
  
  if (opool->tail)
    {
      opool->tail->next = opool->free;
      opool->free = opool->head;
      opool->head = opool->tail = NULL;
    }
}	

void
mu_opool_destroy (mu_opool_t *popool)
{
  struct mu_opool_bucket *p;
  if (popool && *popool)
    {
      mu_opool_t opool = *popool;
      mu_opool_clear (opool);
      for (p = opool->free; p; )
	{
	  struct mu_opool_bucket *next = p->next;
	  free (p);
	  p = next;
	}
      free (opool);
    }
  *popool = NULL;
}

int
mu_opool_append (mu_opool_t opool, const void *str, size_t n)
{
  const char *ptr = str;
  while (n)
    {
      size_t s;
      if (copy_chars (opool, ptr, n, &s))
	return ENOMEM;
      ptr += s;
      n -= s;
    }
  return 0;
}

int
mu_opool_append_char (mu_opool_t opool, char c)
{
  return mu_opool_append (opool, &c, 1);
}	

int
mu_opool_appendz (mu_opool_t opool, const char *str)
{
  return mu_opool_append (opool, str, strlen (str))
         || mu_opool_append_char (opool, 0);
}

size_t
mu_opool_size (mu_opool_t opool)
{
  size_t size = 0;
  struct mu_opool_bucket *p;
  for (p = opool->head; p; p = p->next)
    size += p->level;
  return size;
}

int
mu_opool_coalesce (mu_opool_t opool, size_t *psize)
{
  size_t size;

  if (opool->itr_count)
    return MU_ERR_FAILURE;
  if (opool->head && opool->head->next == NULL)
    size = opool->head->level;
  else {
    struct mu_opool_bucket *bucket;
    struct mu_opool_bucket *p;

    size = mu_opool_size (opool);
	
    bucket = alloc_bucket (opool, size);
    if (!bucket)
      return ENOMEM;
    for (p = opool->head; p; )
      {
	struct mu_opool_bucket *next = p->next;
	memcpy (bucket->buf + bucket->level, p->buf, p->level);
	bucket->level += p->level;
	free (p);
	p = next;
      }
    opool->head = opool->tail = bucket;
  }
  if (psize)
    *psize = size;
  return 0;
}

void *
mu_opool_head (mu_opool_t opool, size_t *psize)
{
  if (*psize) 
    *psize = opool->head ? opool->head->level : 0;
  return opool->head ? opool->head->buf : NULL;
}

void *
mu_opool_finish (mu_opool_t opool, size_t *psize)
{
  if (mu_opool_coalesce (opool, psize))
    return NULL;
  mu_opool_clear (opool);
  return opool->free->buf;
}

int
mu_opool_union (mu_opool_t *pdst, mu_opool_t *psrc)
{
  mu_opool_t src, dst;
  
  if (!psrc)
    return EINVAL;
  if (!*psrc)
    return 0;
  src = *psrc;
  
  if (!pdst)
    return EINVAL;
  if (!*pdst)
    {
      *pdst = src;
      *psrc = NULL;
      return 0;
    }
  else
    dst = *pdst;

  if (dst->tail)
    dst->tail->next = src->head;
  else
    dst->head = src->head;
  dst->tail = src->tail;

  if (src->free)
    {
      struct mu_opool_bucket *p;

      for (p = src->free; p->next; p = p->next)
	;
      p->next = dst->free;
      dst->free = src->free;
    }

  free (src);
  *psrc = NULL;
  return 0;
}


/* Iterator support */
struct opool_iterator
{
  mu_opool_t opool;
  struct mu_opool_bucket *cur;
};

static int
opitr_first (void *owner)
{
  struct opool_iterator *itr = owner;
  itr->cur = itr->opool->head;
  return 0;
}

static int
opitr_next (void *owner)
{
  struct opool_iterator *itr = owner;
  if (itr->cur)
    {
      itr->cur = itr->cur->next;
      return 0;
    }
  return EINVAL;
}

static int
opitr_getitem (void *owner, void **pret, const void **pkey)
{
  struct opool_iterator *itr = owner;
  if (!itr->cur)
    return MU_ERR_NOENT;

  *pret = itr->cur->buf;
  if (pkey)
    *(size_t*) pkey = itr->cur->level;
  return 0;
}

static int
opitr_finished_p (void *owner)
{
  struct opool_iterator *itr = owner;
  return itr->cur == NULL;
}

static int
opitr_curitem_p (void *owner, void *item)
{
  struct opool_iterator *itr = owner;
  return itr->cur && itr->cur->buf == item;
}

static int
opitr_destroy (mu_iterator_t iterator, void *data)
{
  struct opool_iterator *itr = data;
  if (itr->opool->itr_count == 0)
    {
      /* oops! */
      mu_error (_("%s: INTERNAL ERROR: zero reference count"),
		"opool_destroy");
    }
  else
    itr->opool->itr_count--;
  free (data);
  return 0;
}

static int
opitr_data_dup (void **ptr, void *owner)
{
  struct opool_iterator *itr = owner;

  *ptr = malloc (sizeof (struct opool_iterator));
  if (*ptr == NULL)
    return ENOMEM;
  memcpy (*ptr, owner, sizeof (struct opool_iterator));
  itr->opool->itr_count++;
  return 0;
}

int
mu_opool_get_iterator (mu_opool_t opool, mu_iterator_t *piterator)
{
  mu_iterator_t iterator;
  int status;
  struct opool_iterator *itr;

  if (!opool)
    return EINVAL;

  itr = calloc (1, sizeof *itr);
  if (!itr)
    return ENOMEM;
  itr->opool = opool;
  itr->cur = opool->head;
  
  status = mu_iterator_create (&iterator, itr);
  if (status)
    {
      free (itr);
      return status;
    }

  mu_iterator_set_first (iterator, opitr_first);
  mu_iterator_set_next (iterator, opitr_next);
  mu_iterator_set_getitem (iterator, opitr_getitem);
  mu_iterator_set_finished_p (iterator, opitr_finished_p);
  mu_iterator_set_curitem_p (iterator, opitr_curitem_p);
  mu_iterator_set_destroy (iterator, opitr_destroy);
  mu_iterator_set_dup (iterator, opitr_data_dup);

  opool->itr_count++;

  *piterator = iterator;
  return 0;
}




  
  
 
