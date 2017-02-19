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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <mailutils/stream.h>
#include <mailutils/errno.h>

#ifdef _POSIX_MAPPED_FILES
#include <sys/mman.h>

#ifndef MAP_FAILED
# define MAP_FAILED (void*)-1
#endif

struct _mapfile_stream
{
  int fd;
  int flags;
  char *ptr;
  size_t size;
  char* filename;
};

static void
_mapfile_destroy (mu_stream_t stream)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);

  if (mfs->ptr != MAP_FAILED)
    {
      if (mfs->ptr)
	munmap (mfs->ptr, mfs->size);
      close (mfs->fd);
    }
  free (mfs->filename);
  free (mfs);
}

static int
_mapfile_read (mu_stream_t stream, char *optr, size_t osize,
	       mu_off_t offset, size_t *nbytes)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  size_t n = 0;

  if (mfs->ptr == MAP_FAILED)
    return EINVAL;

  if (offset < (mu_off_t)mfs->size)
    {
      n = ((offset + osize) > mfs->size) ? mfs->size - offset :  osize;
      memcpy (optr, mfs->ptr + offset, n);
    }

  if (nbytes)
    *nbytes = n;
  return 0;
}

static int
_mapfile_readline (mu_stream_t stream, char *optr, size_t osize,
		mu_off_t offset, size_t *nbytes)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  char *nl;
  size_t n = 0;

  if (mfs->ptr == MAP_FAILED)
    return EINVAL;

  if (offset < (mu_off_t)mfs->size)
    {
      /* Save space for the null byte.  */
      osize--;
      nl = memchr (mfs->ptr + offset, '\n', mfs->size - offset);
      n = (nl) ? (size_t)(nl - (mfs->ptr + offset) + 1) : mfs->size - offset;
      n = (n > osize)  ? osize : n;
      memcpy (optr, mfs->ptr + offset, n);
      optr[n] = '\0';
    }
  if (nbytes)
    *nbytes = n;
  return 0;
}

static int
_mapfile_write (mu_stream_t stream, const char *iptr, size_t isize,
	    mu_off_t offset, size_t *nbytes)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);

  if (mfs->ptr == MAP_FAILED)
    return EINVAL;

  if (! (mfs->flags & PROT_WRITE))
    return EACCES;

  /* Not recommanded, really.  */
  /* Bigger we have to remmap.  */
  if (mfs->size < (offset + isize))
    {
      if (mfs->ptr && munmap (mfs->ptr, mfs->size) != 0)
	{
	  int err = errno;
	  mfs->ptr = MAP_FAILED;
	  close (mfs->fd);
	  return err;
	}
      if (ftruncate (mfs->fd, offset + isize) != 0)
	return errno;
      mfs->ptr = mmap (0, offset + isize, mfs->flags, MAP_SHARED, mfs->fd, 0);
      if (mfs->ptr == MAP_FAILED)
	{
	  int err = errno;
	  close (mfs->fd);
	  return err;
	}
      mfs->size = offset + isize;
    }

  if (isize)
    memcpy (mfs->ptr + offset, iptr, isize);
  if (nbytes)
    *nbytes = isize;
  return 0;
}

static int
_mapfile_truncate (mu_stream_t stream, mu_off_t len)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  if (mfs->ptr == MAP_FAILED)
    return EINVAL;
  /* Remap.  */
  if (mfs->ptr && munmap (mfs->ptr, mfs->size) != 0)
    {
      int err = errno;
      mfs->ptr = MAP_FAILED;
      close (mfs->fd);
      return err;
    }
  if (ftruncate (mfs->fd, len) != 0)
    return errno;
   mfs->ptr = (len) ? mmap (0, len, mfs->flags, MAP_SHARED, mfs->fd, 0) : NULL;
   if (mfs->ptr == MAP_FAILED)
     {
       int err = errno;
       close (mfs->fd);
       return err;
     }
  mfs->size = len;
  return 0;
}

static int
_mapfile_size (mu_stream_t stream, mu_off_t *psize)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  struct stat stbuf;
  int err = 0;

  if (mfs->ptr == MAP_FAILED)
    return EINVAL;
  if (mfs->ptr && (mfs->flags & PROT_WRITE))
    msync (mfs->ptr, mfs->size, MS_SYNC);
  if (fstat(mfs->fd, &stbuf) != 0)
    return errno;
  if (mfs->size != (size_t)stbuf.st_size)
    {
      if (mfs->ptr)
	err = munmap (mfs->ptr, mfs->size);
      if (err == 0)
	{
	  mfs->size = stbuf.st_size;
	  if (mfs->size)
	    {
	      mfs->ptr = mmap (0, mfs->size, mfs->flags , MAP_SHARED,
			       mfs->fd, 0);
	      if (mfs->ptr == MAP_FAILED)
		err = errno;
	    }
	  else
	    mfs->ptr = NULL;
	}
      else
	err = errno;
    }
  if (err != 0)
    {
      mfs->ptr = MAP_FAILED;
      close (mfs->fd);
      mfs->fd = -1;
    }
  else
    {
      if (psize)
	*psize = stbuf.st_size;
    }
  return err;
}

static int
_mapfile_flush (mu_stream_t stream)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  if (mfs->ptr != MAP_FAILED && mfs->ptr != NULL && (mfs->flags & PROT_WRITE))
    return msync (mfs->ptr, mfs->size, MS_SYNC);
  return 0;
}

static int
_mapfile_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);

  if (pout)
    *pout = NULL;
  
  if (pin)
    *pin = (mu_transport_t) mfs->fd;
  return 0;
}

static int
_mapfile_close (mu_stream_t stream)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  int err = 0;
  if (mfs->ptr != MAP_FAILED)
    {
      if (mfs->ptr && munmap (mfs->ptr, mfs->size) != 0)
	err = errno;
      if (close (mfs->fd) != 0)
	err = errno;
      mfs->ptr = MAP_FAILED;
      mfs->fd = -1;
    }
  return err;
}

static int
_mapfile_open (mu_stream_t stream)
{
  struct _mapfile_stream *mfs = mu_stream_get_owner (stream);
  int mflag, flg;
  struct stat st;
  char* filename = mfs->filename;
  int flags;

  mu_stream_get_flags (stream, &flags);

  /* Close any previous file.  */
  if (mfs->ptr != MAP_FAILED)
    {
      if (mfs->ptr)
	munmap (mfs->ptr, mfs->size);
      mfs->ptr = MAP_FAILED;
    }
  if (mfs->fd != -1)
    {
      close (mfs->fd);
      mfs->fd = -1;
    }
  /* Map the flags to the system equivalent */
  if ((flags & MU_STREAM_WRITE) && (flags & MU_STREAM_READ))
    return EINVAL;
  else if (flags & MU_STREAM_WRITE)
    {
      mflag = PROT_WRITE;
      flg = O_WRONLY;
    }
  else if (flags & MU_STREAM_RDWR)
    {
      mflag = PROT_READ | PROT_WRITE;
      flg = O_RDWR;
    }
  else if (flags & MU_STREAM_CREAT)
    return ENOSYS;
  else /* default */
    {
      mflag = PROT_READ;
      flg = O_RDONLY;
    }

  mfs->fd = open (filename, flg);
  if (mfs->fd < 0)
    return errno;
  if (fstat (mfs->fd, &st) != 0)
    {
      int err = errno;
      close (mfs->fd);
      return err;
    }
  mfs->size = st.st_size;
  if (mfs->size)
    {
      mfs->ptr = mmap (0, mfs->size, mflag , MAP_SHARED, mfs->fd, 0);
      if (mfs->ptr == MAP_FAILED)
	{
	  int err = errno;
	  close (mfs->fd);
	  mfs->ptr = MAP_FAILED;
	  return err;
	}
    }
  else
    mfs->ptr = NULL;
  mfs->flags = mflag;
  mu_stream_set_flags (stream, MU_STREAM_NO_CHECK);
  return 0;
}

#endif /* _POSIX_MAPPED_FILES */

int
mu_mapfile_stream_create (mu_stream_t *stream, const char* filename, int flags)
{
#ifndef _POSIX_MAPPED_FILES
  return ENOSYS;
#else
  struct _mapfile_stream *fs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;
  if (filename == NULL)
    return EINVAL;

  fs = calloc (1, sizeof (struct _mapfile_stream));
  if (fs == NULL)
    return ENOMEM;

  fs->filename = strdup (filename);
  if (!fs->filename)
  {
    free (fs);
    return ENOMEM;
  }

  fs->fd = -1;
  fs->ptr = MAP_FAILED;

  ret = mu_stream_create (stream, flags | MU_STREAM_NO_CHECK, fs);
  if (ret != 0)
    {
      free (fs->filename);
      free (fs);
      return ret;
    }

  mu_stream_set_open (*stream, _mapfile_open, fs);
  mu_stream_set_close (*stream, _mapfile_close, fs);
  mu_stream_set_get_transport2 (*stream, _mapfile_get_transport2, fs);
  mu_stream_set_read (*stream, _mapfile_read, fs);
  mu_stream_set_readline (*stream, _mapfile_readline, fs);
  mu_stream_set_write (*stream, _mapfile_write, fs);
  mu_stream_set_truncate (*stream, _mapfile_truncate, fs);
  mu_stream_set_size (*stream, _mapfile_size, fs);
  mu_stream_set_flush (*stream, _mapfile_flush, fs);
  mu_stream_set_destroy (*stream, _mapfile_destroy, fs);
  return 0;
#endif /* _POSIX_MAPPED_FILES */
}
