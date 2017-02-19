/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2009,
   2010 Free Software Foundation, Inc.

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

/* MH context functions. */
  
#include <mh.h>
#include <sys/types.h>
#include <sys/stat.h>

mh_context_t *
mh_context_create (const char *name, int copy)
{
  mh_context_t *ctx;
  ctx = malloc (sizeof (*ctx));
  if (!ctx)
    mh_err_memory (1);
  if (copy)
    ctx->name = name;
  else
    {
      ctx->name = strdup (name);
      if (!ctx->name)
        mh_err_memory (1);
    }
  ctx->header = NULL;
  return ctx;
}

void
mh_context_destroy (mh_context_t **pctx)
{
  mh_context_t *ctx = *pctx;
  
  free ((char*) ctx->name);
  if (ctx->header)
    mu_header_destroy (&ctx->header, mu_header_get_owner (ctx->header));
  free (ctx);
  *pctx = NULL;
}

void
mh_context_merge (mh_context_t *dst, mh_context_t *src)
{
  if (!dst->header)
    {
      dst->header = src->header;
      src->header = NULL;
    }
  else
    {
      size_t i, count;
      
      mu_header_get_field_count (src->header, &count);
      for (i = 1; i <= count; i++)
	{
	  const char *name = NULL;
	  const char *value = NULL;
	  
	  mu_header_sget_field_name (src->header, i, &name);
	  mu_header_sget_field_value (src->header, i, &value);
	  mu_header_set_value (dst->header, name, value, 1);
	}
    }
}

int 
mh_context_read (mh_context_t *ctx)
{
  int status;
  char *blurb, *p;
  struct stat st;
  FILE *fp;
  char *buf = NULL;
  size_t size = 0;
  
  if (!ctx)
    return MU_ERR_OUT_NULL;
  
  if (stat (ctx->name, &st))
    return errno;

  blurb = malloc (st.st_size);
  if (!blurb)
    return ENOMEM;
  
  fp = fopen (ctx->name, "r");
  if (!fp)
    {
      free (blurb);
      return errno;
    }

  p = blurb;
  while (getline (&buf, &size, fp) > 0)
    {
      char *q;

      for (q = buf; *q && mu_isspace (*q); q++)
	;
      if (!*q || *q == '#')
	continue;
      
      for (q = buf; *q;)
	*p++ = *q++;
    }
  fclose (fp);

  status = mu_header_create (&ctx->header, blurb, p - blurb, NULL);
  free (blurb);

  return status;
}

int 
mh_context_write (mh_context_t *ctx)
{
  mu_stream_t stream;
  char buffer[512];
  size_t off = 0, n;
  FILE *fp;
  
  if (!ctx)
    return MU_ERR_OUT_NULL;

  fp = fopen (ctx->name, "w");
  if (!fp)
    {
      mu_error (_("cannot open context file %s: %s"),
		ctx->name, strerror (errno));
      return MU_ERR_FAILURE;
    }
  
  mu_header_get_stream (ctx->header, &stream);

  while (mu_stream_read (stream, buffer, sizeof buffer - 1, off, &n) == 0
	 && n != 0)
    {
      buffer[n] = '\0';
      fprintf (fp, "%s", buffer);
      off += n;
    }

  fclose (fp);
  return 0;
}

const char *
mh_context_get_value (mh_context_t *ctx, const char *name, const char *defval)
{
  const char *p;

  if (!ctx || mu_header_sget_value (ctx->header, name, &p))
    p = defval; 
  return p;
}

int
mh_context_set_value (mh_context_t *ctx, const char *name, const char *value)
{
  if (!ctx)
    return EINVAL;
  if (!ctx->header)
    {
      int rc;
      if ((rc = mu_header_create (&ctx->header, NULL, 0, NULL)) != 0)
	{
	  mu_error (_("cannot create context %s: %s"),
		    ctx->name,
		    mu_strerror (rc));
	  return 1;
	}
    }
  return mu_header_set_value (ctx->header, name, value, 1);
}

int
mh_context_iterate (mh_context_t *ctx, mh_context_iterator fp, void *data)
{
  size_t i, nfields;
  int rc = 0;
  
  if (!ctx)
    return EINVAL;
  mu_header_get_field_count (ctx->header, &nfields);
  for (i = 1; i <= nfields && rc == 0; i++)
    {
      const char *name, *value;
      
      mu_header_sget_field_name (ctx->header, i, &name);
      mu_header_sget_field_value (ctx->header, i, &value);
      rc = fp (name, value, data);
    }

  return rc;
}
