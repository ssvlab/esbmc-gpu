/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2004, 2007, 2010 Free Software
   Foundation, Inc.

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

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

#include <sys/types.h>

#include <mailutils/stream.h>
#include <mailutils/errno.h>

#undef min
#define min(a,b) ((a) < (b) ? (a) : (b))

#define MU_STREAM_MEMORY_BLOCKSIZE 128

struct _memory_stream
{
  char *filename;
  char *ptr;
  size_t size;
  size_t capacity;
};

static void
_memory_destroy (mu_stream_t stream)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  if (mfs && mfs->ptr != NULL)
    free (mfs->ptr);
  if (mfs->filename)
    free (mfs->filename);
  free (mfs);
}

static int
_memory_read (mu_stream_t stream, char *optr, size_t osize,
	      mu_off_t offset, size_t *nbytes)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  size_t n = 0;
  if (mfs->ptr != NULL && ((size_t)offset <= mfs->size))
    {
      n = ((offset + osize) > mfs->size) ? mfs->size - offset :  osize;
      memcpy (optr, mfs->ptr + offset, n);
    }
  if (nbytes)
    *nbytes = n;
  return 0;
}

static int
_memory_readline (mu_stream_t stream, char *optr, size_t osize,
		  mu_off_t offset, size_t *nbytes)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  char *nl;
  size_t n = 0;
  if (mfs->ptr && ((size_t)offset < mfs->size))
    {
      /* Save space for the null byte.  */
      osize--;
      nl = memchr (mfs->ptr + offset, '\n', mfs->size - offset);
      n = (nl) ? (size_t)(nl - (mfs->ptr + offset) + 1) : mfs->size - offset;
      n = min (n, osize);
      memcpy (optr, mfs->ptr + offset, n);
      optr[n] = '\0';
    }
  if (nbytes)
    *nbytes = n;
  return 0;
}

static int
_memory_write (mu_stream_t stream, const char *iptr, size_t isize,
	       mu_off_t offset, size_t *nbytes)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);

  /* Bigger we have to realloc.  */
  if (mfs->capacity < ((size_t)offset + isize))
    {
      /* Realloc by fixed blocks of 128.  */
      size_t newsize = MU_STREAM_MEMORY_BLOCKSIZE *
	(((offset + isize)/MU_STREAM_MEMORY_BLOCKSIZE) + 1);
      char *tmp =  realloc (mfs->ptr, newsize);
      if (tmp == NULL)
	return ENOMEM;
      mfs->ptr = tmp;
      mfs->capacity = newsize;
    }

  mfs->size = offset + isize;
  memcpy (mfs->ptr + offset, iptr, isize);
  if (nbytes)
    *nbytes = isize;
  return 0;
}

static int
_memory_truncate (mu_stream_t stream, mu_off_t len)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);

  if (len > (mu_off_t)mfs->size)
    {
      char *tmp = realloc (mfs->ptr, len);
      if (tmp == NULL)
	return ENOMEM;
      mfs->ptr = tmp;
      mfs->capacity = len;
    }
  mfs->size = len;
  return 0;
}

static int
_memory_size (mu_stream_t stream, mu_off_t *psize)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  if (psize)
    *psize = mfs->size;
  return 0;
}

static int
_memory_close (mu_stream_t stream)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  if (mfs->ptr)
    free (mfs->ptr);
  mfs->ptr = NULL;
  mfs->size = 0;
  mfs->capacity = 0;
  return 0;
}

static int
_memory_open (mu_stream_t stream)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  int status = 0;

  /* Close any previous file.  */
  if (mfs->ptr)
    free (mfs->ptr);
  mfs->ptr = NULL;
  mfs->size = 0;
  mfs->capacity = 0;

  /* Initialize the data with file contents, if a filename was provided. */
  if (mfs->filename)
    {
      struct stat statbuf;
      if (stat (mfs->filename, &statbuf) == 0)
        {
          mfs->ptr = calloc (statbuf.st_size, 1);
          if (mfs->ptr)
            {
              FILE *fp;
              mfs->capacity = statbuf.st_size;
              mfs->size = statbuf.st_size;
              fp = fopen (mfs->filename, "r");
              if (fp)
                {
                  size_t r = fread (mfs->ptr, mfs->size, 1, fp);
                  if (r != mfs->size)
                    status = EIO;
                  fclose (fp);
                }
              else
                status = errno;
              if (status != 0)
                {
                  free (mfs->ptr);
                  mfs->ptr = NULL;
                  mfs->capacity = 0;
                  mfs->size = 0;
                }
            }
          else
            status = ENOMEM;
        }
      else
        status = EIO;
    }
  return status;
}

static int
_memory_get_transport2 (mu_stream_t stream,
			mu_transport_t *pin, mu_transport_t *pout)
{
  struct _memory_stream *mfs = mu_stream_get_owner (stream);
  *pin = mfs->ptr;
  if (pout)
    *pout = mfs->ptr;
  return 0;
}

int
mu_memory_stream_create (mu_stream_t * stream, const char *filename, int flags)
{
  struct _memory_stream *mfs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  mfs = calloc (1, sizeof (*mfs));

  if (mfs == NULL)
    return ENOMEM;

  if (filename)
    {
      mfs->filename = strdup (filename);
      if (!mfs->filename)
	{
	  free (mfs);
	  return ENOMEM;
	}
    }

  mfs->ptr = NULL;
  mfs->size = 0;

  ret = mu_stream_create (stream, flags | MU_STREAM_NO_CHECK, mfs);
  if (ret != 0)
    {
      free (mfs->filename);
      free (mfs);

      return ret;
    }

  mu_stream_set_open (*stream, _memory_open, mfs);
  mu_stream_set_close (*stream, _memory_close, mfs);
  mu_stream_set_read (*stream, _memory_read, mfs);
  mu_stream_set_readline (*stream, _memory_readline, mfs);
  mu_stream_set_write (*stream, _memory_write, mfs);
  mu_stream_set_truncate (*stream, _memory_truncate, mfs);
  mu_stream_set_size (*stream, _memory_size, mfs);
  mu_stream_set_destroy (*stream, _memory_destroy, mfs);
  mu_stream_set_get_transport2 (*stream, _memory_get_transport2, mfs);
  
  return 0;
}
