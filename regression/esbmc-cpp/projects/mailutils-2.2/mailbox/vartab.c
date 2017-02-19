/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2007, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; If not, see
   <http://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <mailutils/assoc.h>
#include <mailutils/stream.h>
#include <mailutils/errno.h>
#include <mailutils/iterator.h>
#include <mailutils/vartab.h>

struct _mu_vartab
{
  mu_assoc_t assoc;
  mu_stream_t stream;
  char *buf;
  size_t bufsize;
};

#define MU_VARDEFN_STATIC 0x1
#define MU_VARDEFN_FUNC   0x2

struct vardefn
{
  int flags;
  char *value;
  mu_var_expansion_fp fun;
  mu_var_free_fp free;
  void *data;
};

int
mu_vartab_create (mu_vartab_t *pvar)
{
  int rc;
  struct _mu_vartab *p = calloc (1, sizeof *p);
  if (!p)
    return ENOMEM;
  rc = mu_assoc_create (&p->assoc, sizeof (struct vardefn), 0);
  if (rc)
    {
      free (p);
      return rc;
    }
  *pvar = p;
  return 0;
}

static void
vardefn_free (struct vardefn *vd)
{
  if (vd->value)
    {
      if (vd->free)
	vd->free (vd->data, vd->value);
      else if (!(vd->flags & MU_VARDEFN_STATIC))
	free (vd->value);
    }
  memset (vd, 0, sizeof vd);
}

int
mu_vartab_destroy (mu_vartab_t *pvar)
{
  int rc;
  mu_vartab_t var = *pvar;
  mu_iterator_t itr;
  
  if (!var)
    return EINVAL;
  rc = mu_assoc_get_iterator (var->assoc, &itr);
  if (rc)
    return rc;
  for (mu_iterator_first (itr); !mu_iterator_is_done (itr);
       mu_iterator_next (itr))
    {
      struct vardefn *vd;
      mu_iterator_current (itr, (void**)&vd);
      vardefn_free (vd);
    }
  mu_iterator_destroy (&itr);

  mu_assoc_destroy (&var->assoc);
  mu_stream_destroy (&var->stream, NULL);
  free (var->buf);
  free (var);
  *pvar = NULL;
  return 0;
}

int
mu_vartab_define (mu_vartab_t var, const char *name, const char *value,
		  int isstatic)
{
  int rc;
  struct vardefn *vd;

  if (!var)
    return EINVAL;
  rc = mu_assoc_ref_install (var->assoc, name, (void **) &vd);
  if (rc == MU_ERR_EXISTS)
    vardefn_free (vd);
  else if (rc != 0)
    return rc;

  if (isstatic)
    {
      vd->flags = MU_VARDEFN_STATIC;
      vd->value = (char*) value;
    }
  else
    {
      vd->flags = 0;
      vd->value = strdup (value);
      if (!vd->value)
	return ENOMEM;
    }
  return 0;
}

int
mu_vartab_define_exp (mu_vartab_t var, const char *name,
		      mu_var_expansion_fp fun, mu_var_free_fp free,
		      void *data)
{
  int rc;
  struct vardefn *vd;

  if (!var)
    return EINVAL;
  rc = mu_assoc_ref_install (var->assoc, name, (void **) &vd);
  if (rc == MU_ERR_EXISTS)
    vardefn_free (vd);
  else if (rc != 0)
    return rc;

  vd->flags = MU_VARDEFN_FUNC;
  vd->fun = fun;
  vd->free = free;
  vd->data = data;
  return 0;
}

int
mu_vartab_count (mu_vartab_t vt, size_t *pcount)
{
  if (!vt)
    return EINVAL;
  return mu_assoc_count (vt->assoc, pcount);
}

static int
vardefn_expand (const char *name, struct vardefn *vd, const char **pvalue)
{
  if (!vd->value)
    {
      if (vd->fun)
	{
	  int rc = vd->fun (name, vd->data, &vd->value);
	  if (rc)
	    return rc;
	}
      else
	return EINVAL;
    }
  *pvalue = vd->value;
  return 0;
}

int
mu_vartab_getvar (mu_vartab_t vt, const char *name, const char **pvalue)
{
  struct vardefn *vdefn;
  
  if (!vt)
    return EINVAL;
  vdefn = mu_assoc_ref (vt->assoc, name);
  if (!vdefn)
    return MU_ERR_NOENT;
  return vardefn_expand (name, vdefn, pvalue);
}

static char *
copy_name (mu_vartab_t vt, const char *name, size_t len)
{
  if (len + 1 > vt->bufsize)
    {
      char *p = realloc (vt->buf, len + 1);
      if (!p)
	return NULL;
      vt->buf = p;
      vt->bufsize = len + 1;
    }
  memcpy (vt->buf, name, len);
  vt->buf[len] = 0;
  return vt->buf;
}

int
mu_vartab_expand (mu_vartab_t vt, const char *str, char **pres)
{
  int rc;
  mu_off_t size;
  const char *p;
  
  if (!vt)
    return EINVAL;
  if (!vt->stream)
    {
      rc = mu_memory_stream_create (&vt->stream, NULL, MU_STREAM_NO_CHECK);
      if (rc)
	return rc;
      rc = mu_stream_open (vt->stream);
    }
  else
    mu_stream_truncate (vt->stream, 0);
  mu_stream_seek (vt->stream, 0, SEEK_SET);

  for (p = str; *p; )
    {
      if (*p == '$')
	{
	  switch (*++p)
	    {
	    case '$':
	      mu_stream_sequential_write (vt->stream, str, p - str);
	      str = p + 1;
	      p = str + 1;
	      break;
	      
	    case '{':
	      {
		const char *e = strchr (p + 1, '}');
		if (e)
		  {
		    const char *pvalue;
		    size_t len = e - p - 1;
		    char *name = copy_name (vt, p + 1, len);
		    rc = mu_vartab_getvar (vt, name, &pvalue);
		    if (rc == 0)
		      {
			mu_stream_sequential_write (vt->stream, str,
						    p - str - 1);
			mu_stream_sequential_write (vt->stream, pvalue,
						    strlen (pvalue));
			str = e + 1;
			p = str + 1;
		      }
		    else if (rc == MU_ERR_NOENT)
		      p = e + 1;
		    else
		      return rc;
		  }
		else
		  p++;
	      }
	      break;
	      
	    default:
	      {
		char *name = copy_name (vt, p, 1);
		const char *pvalue;
		rc = mu_vartab_getvar (vt, name, &pvalue);
		if (rc == 0)
		  {
		    mu_stream_sequential_write (vt->stream, str,
						p - str - 1);
		    mu_stream_sequential_write (vt->stream, pvalue,
						strlen (pvalue));
		    str = p + 1;
		    p = str + 1;
		  }
		else if (rc == MU_ERR_NOENT)
		  p++;
		else
		  return rc;
	      }
	      break;
	    }
	}
      else if (*p == '%')
	{
	  /* allow `%' as prefix for single-character entities, for
	     compatibility with v. prior to 1.2.91 */
	  if (*++p == '%')
	    {
	      mu_stream_sequential_write (vt->stream, str, p - str);
	      str = p + 1;
	      p = str + 1;
	    }
	  else
	    {
	      char *name = copy_name (vt, p, 1);
	      const char *pvalue;
	      rc = mu_vartab_getvar (vt, name, &pvalue);
	      if (rc == 0)
		{
		  mu_stream_sequential_write (vt->stream, str,
					      p - str - 1);
		  mu_stream_sequential_write (vt->stream, pvalue,
					      strlen (pvalue));
		  str = p + 1;
		  p = str + 1;
		}
	      else if (rc == MU_ERR_NOENT)
		p++;
	      else
		return rc;
	    }
	}
      else
	p++;
    }
  
  if (p > str)
    mu_stream_sequential_write (vt->stream, str, p - str);

  mu_stream_size (vt->stream, &size);
  *pres = malloc (size + 1);
  if (!*pres)
    return ENOMEM;
  mu_stream_read (vt->stream, *pres, size, 0, NULL);
  (*pres)[size] = 0;
  return 0;
}
  
     


  
		  
  
