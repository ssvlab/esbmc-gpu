/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2009,
   2010 Free Software Foundation, Inc.

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
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "mailutils/libargp.h"


/* ************************************************************************* */
/* Capability array and auxiliary functions.                                 */
/* ************************************************************************* */

#define MU_MAX_CAPA 24

struct argp_capa {
  char *capability;
  struct argp_child *child;
} mu_argp_capa[MU_MAX_CAPA] = {
  {NULL,}
};

int
mu_register_argp_capa (const char *name, struct argp_child *child)
{
  int i;
  
  for (i = 0; i < MU_MAX_CAPA; i++)
    if (mu_argp_capa[i].capability == NULL)
      {
	mu_argp_capa[i].capability = strdup (name);
	mu_argp_capa[i].child = child;
	return 0;
      }
  return 1;
}

static struct argp_capa *
find_capa (const char *name)
{
  int i;
  for (i = 0; mu_argp_capa[i].capability; i++)
    if (strcmp (mu_argp_capa[i].capability, name) == 0)
      return &mu_argp_capa[i];
  return NULL;
}

static struct argp *
mu_build_argp (const struct argp *template, char **capa)
{
  int n;
  int nchild;
  struct argp_child *ap;
  const struct argp_option *opt;
  struct argp *argp;
  int group = 0;

  /* Count the capabilities. */
  for (n = 0; capa && capa[n]; n++)
    ;
  if (template->children)
    for (; template->children[n].argp; n++)
      ;

  ap = calloc (n + 1, sizeof (*ap));
  if (!ap)
    {
      mu_error (_("not enough memory"));
      abort ();
    }

  /* Copy the template's children. */
  nchild = 0;
  if (template->children)
    for (n = 0; template->children[n].argp; n++, nchild++)
      ap[nchild] = template->children[n];

  /* Find next group number */
  for (opt = template->options;
       opt && ((opt->name && opt->key) || opt->doc); opt++)
    if (opt->group > group)
      group = opt->group;

  group++;
    
  /* Append any capabilities to the children or options, as appropriate. */
  for (n = 0; capa && capa[n]; n++)
    {
      struct argp_capa *cp = find_capa (capa[n]);
      if (cp)
	{
	  ap[nchild] = *cp->child;
	  ap[nchild].group = group++;
	  nchild++;
	}
    }
  ap[nchild].argp = NULL;

  /* Copy the template, and give it the expanded children. */
  argp = malloc (sizeof (*argp));
  if (!argp)
    {
      mu_error (_("not enough memory"));
      abort ();
    }

  memcpy (argp, template, sizeof (*argp));

  argp->children = ap;

  return argp;
}

struct cap_buf
{
  char **capa;
  size_t numcapa;
  size_t maxcapa;
};

static void
cap_buf_init (struct cap_buf *bp)
{
  bp->numcapa = 0;
  bp->maxcapa = 2;
  bp->capa = calloc (bp->maxcapa, sizeof bp->capa[0]);
  if (!bp->capa)
    {
      mu_error ("%s", mu_strerror (errno));
      abort ();
    }
  bp->capa[0] = NULL;
}

static void
cap_buf_add (struct cap_buf *bp, char *str)
{
  if (bp->numcapa == bp->maxcapa)
    {
      bp->maxcapa *= 2;
      bp->capa = realloc (bp->capa, bp->maxcapa * sizeof bp->capa[0]);
      if (!bp->capa)
	{
	  mu_error ("%s", mu_strerror (errno));
	  abort ();
	}
    }
  bp->capa[bp->numcapa] = str;
  if (str)
    bp->numcapa++;
}

static void
cap_buf_free (struct cap_buf *bp)
{
  free (bp->capa);
}

static int
argp_reg_action (void *item, void *data)
{
  struct cap_buf *bp = data;
  cap_buf_add (bp, item);
  return 0;
}

struct argp *
mu_argp_build (const struct argp *init_argp, char ***pcapa)
{
  struct cap_buf cb;
  struct argp *argp;

  cap_buf_init (&cb);
  mu_gocs_enumerate (argp_reg_action, &cb);
  cap_buf_add (&cb, NULL);
  mu_libargp_init ();
  argp = mu_build_argp (init_argp, cb.capa);
  if (pcapa)
    *pcapa = cb.capa;
  else
    cap_buf_free (&cb);
  return argp;
}

void
mu_argp_done (struct argp *argp)
{
  free ((void*) argp->children);
  free ((void*) argp);
}
