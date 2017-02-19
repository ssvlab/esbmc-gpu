/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.

   GNU Mailutils is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GNU Mailutils is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GNU Mailutils; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301 USA */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <mailutils/types.h>
#include <mailutils/assoc.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/iterator.h>
#include <mailutils/mutil.h>
#include <mailutils/cstr.h>
#include <iterator0.h>

/* |hash_size| defines a sequence of symbol table sizes. These are prime
   numbers, the distance between each pair of them grows exponentially,
   starting from 64. Hardly someone will need more than 16411 symbols, and
   even if someone will, it is easy enough to add more numbers to the
   sequence. */

static unsigned int hash_size[] = {
    37,   101,  229,  487, 1009, 2039, 4091, 8191, 16411
};

/* |max_rehash| keeps the number of entries in |hash_size| table. */
static unsigned int max_rehash = sizeof (hash_size) / sizeof (hash_size[0]);

struct _mu_assoc_elem
{
  char *name;
  char data[1];
};

struct _mu_assoc
{
  int flags;
  unsigned int hash_num;  /* Index to hash_size table */
  size_t elsize;          /* Size of an element */
  void *tab;
  mu_assoc_free_fn free;
  mu_iterator_t itr;
};

struct _mu_assoc_elem_align
{
  char c;
  struct _mu_assoc_elem x;
};

#define __ASSOC_ELEM_ALIGNMENT (mu_offsetof(struct _mu_assoc_elem_align, x))

#define __ASSOC_ELEM_SIZE(a) \
   ((a)->elsize + mu_offsetof(struct _mu_assoc_elem, data))
#define __ASSOC_ALIGN(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define ASSOC_ELEM_SIZE(a) \
   __ASSOC_ALIGN(__ASSOC_ELEM_SIZE(a),__ASSOC_ELEM_ALIGNMENT)

#define __ASSOC_ELEM(a,p,n) \
 ((struct _mu_assoc_elem*) ((char*) (p) + ASSOC_ELEM_SIZE (a) * n))

#define ASSOC_ELEM(a,n) __ASSOC_ELEM(a,(a)->tab,n)

#define ASSOC_ELEM_INDEX(a,e) \
 (((char*)(e) - (char*)(a)->tab) / ASSOC_ELEM_SIZE (a))


static unsigned
hash (const char *name, unsigned long hash_num)
{
  unsigned i;
	
  for (i = 0; *name; name++)
    {
      i <<= 1;
      i ^= *(unsigned char*) name;
    }
  return i % hash_size[hash_num];
};

static int
assoc_lookup_or_install (struct _mu_assoc_elem **elp,
			 mu_assoc_t assoc, const char *name, int *install);

static int
assoc_rehash (mu_assoc_t assoc)
{
  void *old_tab = assoc->tab;
  void *new_tab;
  unsigned int i;
  unsigned int hash_num = assoc->hash_num + 1;
  
  if (hash_num >= max_rehash)
      return MU_ERR_BUFSPACE;

  new_tab = calloc (hash_size[hash_num], ASSOC_ELEM_SIZE (assoc));
  assoc->tab = new_tab;
  if (old_tab)
    {
      assoc->hash_num = hash_num;
      for (i = 0; i < hash_size[hash_num-1]; i++)
	{
	  struct _mu_assoc_elem *elt = __ASSOC_ELEM (assoc, old_tab, i);
	  if (elt->name)
	    {
	      int tmp;
	      struct _mu_assoc_elem *newp;
	      
	      int rc = assoc_lookup_or_install (&newp, assoc, elt->name, &tmp);
	      if (rc)
		return rc;
	      newp->name = elt->name;
	      memcpy(newp->data, elt->data, assoc->elsize);
	    }
	}
      free (old_tab);
    }
  return 0;
}

static void
assoc_free_elem (mu_assoc_t assoc, struct _mu_assoc_elem *elem)
{
  if (assoc->free)
    assoc->free (elem->data);
  if (!(assoc->flags & MU_ASSOC_COPY_KEY))
    free (elem->name);
}

static int
assoc_remove (mu_assoc_t assoc, struct _mu_assoc_elem *elem)
{
  unsigned int i, j, r;

  if (!(ASSOC_ELEM (assoc, 0) <= elem
	&& elem < ASSOC_ELEM (assoc, hash_size[assoc->hash_num])))
    return EINVAL;

  assoc_free_elem (assoc, elem);
  
  for (i = ASSOC_ELEM_INDEX (assoc, elem);;)
    {
      struct _mu_assoc_elem *p = ASSOC_ELEM (assoc, i);
      p->name = NULL;
      j = i;

      do
	{
	  if (++i >= hash_size[assoc->hash_num])
	    i = 0;
	  p = ASSOC_ELEM (assoc, i);
	  if (!p->name)
	    return 0;
	  r = hash (p->name, assoc->hash_num);
	}
      while ((j < r && r <= i) || (i < j && j < r) || (r <= i && i < j));

      if (j != i)
	      memcpy (ASSOC_ELEM (assoc, j), ASSOC_ELEM (assoc, i),
		      assoc->elsize);
    }
  return 0;
}

#define name_cmp(assoc,a,b) (((assoc)->flags & MU_ASSOC_ICASE) ? \
                             mu_c_strcasecmp(a,b) : strcmp(a,b))

static int
assoc_lookup_or_install (struct _mu_assoc_elem **elp,
			 mu_assoc_t assoc, const char *name, int *install)
{
  int rc;
  unsigned i, pos;
  struct _mu_assoc_elem *elem;
  
  if (!assoc->tab)
    {
      if (install)
	{
	  rc = assoc_rehash (assoc);
	  if (rc)
	    return rc;
	}
      else
	return MU_ERR_NOENT;
    }

  pos = hash (name, assoc->hash_num);

  for (i = pos; (elem = ASSOC_ELEM (assoc, i))->name;)
    {
      if (name_cmp (assoc, elem->name, name) == 0)
	{
	  if (install)
	    *install = 0;
	  *elp = elem;
	  return 0;
	}
      
      if (++i >= hash_size[assoc->hash_num])
	i = 0;
      if (i == pos)
	break;
    }

  if (!install)
    return MU_ERR_NOENT;
  
  if (elem->name == NULL)
    {
      *install = 1;
      if (assoc->flags & MU_ASSOC_COPY_KEY)
	elem->name = (char *) name;
      else
	{
	  elem->name = strdup (name);
	  if (!elem->name)
	    return ENOMEM;
	}
      *elp = elem;
      return 0; 
    }

  if ((rc = assoc_rehash (assoc)) != 0)
    return rc;

  return assoc_lookup_or_install (elp, assoc, name, install);
}

int
mu_assoc_create (mu_assoc_t *passoc, size_t elsize, int flags)
{
  mu_assoc_t assoc = calloc (1, sizeof (*assoc));
  if (!assoc)
    return ENOMEM;
  assoc->flags = flags;
  assoc->elsize = elsize;
  *passoc = assoc;
  return 0;
}

void
mu_assoc_clear (mu_assoc_t assoc)
{
  unsigned i, hs;
  
  if (!assoc || !assoc->tab)
    return;

  hs = hash_size[assoc->hash_num];
  for (i = 0; i < hs; i++)
    {
      struct _mu_assoc_elem *elem = ASSOC_ELEM (assoc, i);
      if (elem->name)
	{
	  assoc_free_elem (assoc, elem);
	  elem->name = NULL;
	}
    }
}

void
mu_assoc_destroy (mu_assoc_t *passoc)
{
  mu_assoc_t assoc;
  if (passoc && (assoc = *passoc) != NULL)
    {
      mu_assoc_clear (assoc);
      free (assoc->tab);
      free (assoc);
      *passoc = NULL;
    }
}

int
mu_assoc_set_free (mu_assoc_t assoc, mu_assoc_free_fn fn)
{
  if (!assoc)
    return EINVAL;
  assoc->free = fn;
  return 0;
}

void *
mu_assoc_ref (mu_assoc_t assoc, const char *name)
{
  int rc;
  static struct _mu_assoc_elem *elp;

  if (!assoc || !name)
    return NULL;
  
  rc = assoc_lookup_or_install (&elp, assoc, name, NULL);
  if (rc == 0)
    return elp->data;
  return NULL;
}

int
mu_assoc_install (mu_assoc_t assoc, const char *name, void *value)
{
  int rc;
  int inst;
  static struct _mu_assoc_elem *elp;

  if (!assoc || !name)
    return EINVAL;

  rc = assoc_lookup_or_install (&elp, assoc, name, &inst);
  if (rc)
    return rc;
  if (!inst)
    return MU_ERR_EXISTS;
  memcpy (elp->data, value, assoc->elsize);
  return 0;
}

int
mu_assoc_ref_install (mu_assoc_t assoc, const char *name, void **pval)
{
  int rc;
  int inst;
  static struct _mu_assoc_elem *elp;

  if (!assoc || !name)
    return EINVAL;

  rc = assoc_lookup_or_install (&elp, assoc, name, &inst);
  if (rc)
    return rc;
  *pval = elp->data;
  return inst ? 0 : MU_ERR_EXISTS;
}  

int
mu_assoc_remove (mu_assoc_t assoc, const char *name)
{
  int rc;
  static struct _mu_assoc_elem *elem;

  if (!assoc || !name)
    return EINVAL;
  rc = assoc_lookup_or_install (&elem, assoc, name, NULL);
  if (rc)
    return rc;
  return assoc_remove (assoc, elem);
}

#define OFFSET(s,f) (size_t)(&((s*)0)->f)

int
mu_assoc_remove_ref (mu_assoc_t assoc, void *data)
{
  struct _mu_assoc_elem *elem;

  elem = (struct _mu_assoc_elem *) ((char*)data -
				    OFFSET(struct _mu_assoc_elem, data));
  return assoc_remove (assoc, elem);
}


/* Iterator interface */

struct assoc_iterator
{
  mu_assoc_t assoc;
  unsigned start;
  unsigned index;
};

static int
first (void *owner)
{
  struct assoc_iterator *itr = owner;
  mu_assoc_t assoc = itr->assoc;
  unsigned hash_max = hash_size[assoc->hash_num];
  unsigned i;
  
  for (i = 0; i < hash_max; i++)
    if ((ASSOC_ELEM (assoc, i))->name)
      break;
  itr->index = i;
  return 0;
}

static int
next (void *owner)
{
  struct assoc_iterator *itr = owner;
  mu_assoc_t assoc = itr->assoc;
  unsigned hash_max = hash_size[assoc->hash_num];
  unsigned i;
  
  for (i = itr->index + 1; i < hash_max; i++)
    if ((ASSOC_ELEM (assoc, i))->name)
      break;

  itr->index = i;
  return 0;
}

static int
getitem (void *owner, void **pret, const void **pkey)
{
  struct assoc_iterator *itr = owner;
  struct _mu_assoc_elem *elem;

  if (itr->index >= hash_size[itr->assoc->hash_num])
    return EINVAL;
  elem = ASSOC_ELEM (itr->assoc, itr->index);
  *pret = elem->data;
  if (pkey)
    *pkey = elem->name;
  return 0;
}

static int
finished_p (void *owner)
{
  struct assoc_iterator *itr = owner;
  return itr->index >= hash_size[itr->assoc->hash_num];
}

static int
destroy (mu_iterator_t iterator, void *data)
{
  struct assoc_iterator *itr = data;
  mu_iterator_detach (&itr->assoc->itr, iterator);
  free (data);
  return 0;
}

static int
curitem_p (void *owner, void *item)
{
  struct assoc_iterator *itr = owner;
  mu_assoc_t assoc = itr->assoc;
  struct _mu_assoc_elem *elem = ASSOC_ELEM (assoc, itr->index);
  
  return elem == item;
}

static int
assoc_data_dup (void **ptr, void *owner)
{
  *ptr = malloc (sizeof (struct assoc_iterator));
  if (*ptr == NULL)
    return ENOMEM;
  memcpy (*ptr, owner, sizeof (struct assoc_iterator));
  return 0;
}

int
mu_assoc_get_iterator (mu_assoc_t assoc, mu_iterator_t *piterator)
{
  mu_iterator_t iterator;
  int status;
  struct assoc_iterator *itr;

  if (!assoc)
    return EINVAL;

  itr = calloc (1, sizeof *itr);
  if (!itr)
    return ENOMEM;
  itr->assoc = assoc;
  itr->index = 0;

  status = mu_iterator_create (&iterator, itr);
  if (status)
    {
      free (itr);
      return status;
    }

  mu_iterator_set_first (iterator, first);
  mu_iterator_set_next (iterator, next);
  mu_iterator_set_getitem (iterator, getitem);
  mu_iterator_set_finished_p (iterator, finished_p);
  mu_iterator_set_curitem_p (iterator, curitem_p);
  mu_iterator_set_destroy (iterator, destroy);
  mu_iterator_set_dup (iterator, assoc_data_dup);
  
  mu_iterator_attach (&assoc->itr, iterator);

  *piterator = iterator;
  return 0;
}  



int
mu_assoc_count (mu_assoc_t assoc, size_t *pcount)
{
  mu_iterator_t itr;
  int rc;
  size_t count = 0;
  
  if (!assoc || !pcount)
    return EINVAL;
  rc = mu_assoc_get_iterator (assoc, &itr);
  if (rc)
    return rc;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    count++;
  mu_iterator_destroy (&itr);
  *pcount = count;
  return 0;
}

