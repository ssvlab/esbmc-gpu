/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008,
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

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <syslog.h>
#include <signal.h>

#include <mailutils/stream.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/argcv.h>
#include <mailutils/nls.h>
#include <mailutils/list.h>
#include <mailutils/mutil.h>

struct _file_stream
{
  FILE *file;
  mu_off_t offset;
  int tempfile;
  char *filename;
  /* The following three members are used for stdio streams only. */
  int size_computed;
  mu_stream_t cache;
  mu_off_t size;
};

static void
_file_destroy (mu_stream_t stream)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);

  if (fs->filename)
    free (fs->filename);

  if (fs->cache)
    mu_stream_destroy (&fs->cache, mu_stream_get_owner (fs->cache));
  free (fs);
}

static int
_file_read (mu_stream_t stream, char *optr, size_t osize,
	    mu_off_t offset, size_t *nbytes)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  size_t n;
  int err = 0;

  if (!fs->file)
    {
      if (nbytes)
	*nbytes = 0;
      return 0;
    }

  if (fs->offset != offset)
    {
      if (fseeko (fs->file, offset, SEEK_SET) != 0)
	return errno;
      fs->offset = offset;
    }

  if (feof (fs->file))
    {
      if (nbytes)
	*nbytes = 0;
      return 0;
    }
  
  n = fread (optr, sizeof(char), osize, fs->file);
  if (n == 0)
    {
      if (ferror(fs->file))
	err = errno;
    }
  else
    fs->offset += n;

  if (nbytes)
    *nbytes = n;
  return err;
}

static int
_file_readline (mu_stream_t stream, char *optr, size_t osize,
		mu_off_t offset, size_t *nbytes)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  size_t n = 0;
  int err = 0;

  if (!fs->file)
    {
      optr[0] = 0;
      if (nbytes)
	*nbytes = 0;
      return 0;
    }

  if (fs->offset != offset)
    {
      if (fseeko (fs->file, offset, SEEK_SET) != 0)
	return errno;
      fs->offset = offset;
    }

  if (feof (fs->file))
    {
      if (nbytes)
	*nbytes = 0;
      return 0;
    }

  if (fgets (optr, osize, fs->file) != NULL)
    {
      char *tmp = optr;
      while (*tmp) tmp++; /* strlen(optr) */
      n = tmp - optr;
      /* !!!!! WTF ??? */
      if (n == 0)
	n++;
      else
	fs->offset += n;
    }
  else
    {
      if (ferror (fs->file))
	err = errno;
    }

  optr[n] = 0;
  if (nbytes)
    *nbytes = n;
  return err;
}

static int
_file_write (mu_stream_t stream, const char *iptr, size_t isize,
	     mu_off_t offset, size_t *nbytes)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  size_t n;
  int err = 0;

  if (!fs->file)
    {
      if (nbytes)
	*nbytes = 0;
      return 0;
    }

  if (fs->offset != offset)
    {
      if (fseeko (fs->file, offset, SEEK_SET) != 0)
	return errno;
      fs->offset = offset;
    }

  n = fwrite (iptr, sizeof(char), isize, fs->file);
  if (n != isize)
    {
      if (feof (fs->file) == 0)
	err = EIO;
      else if (n == 0)
	err = ENOSPC; 
      clearerr(fs->file);
      n = 0;
    }
  else
    fs->offset += n;

  if (nbytes)
    *nbytes = n;
  return err;
}

static int
_stdin_file_read (mu_stream_t stream, char *optr, size_t osize,
		  mu_off_t offset, size_t *pnbytes)
{
  int status = 0;
  size_t nbytes;
  struct _file_stream *fs = mu_stream_get_owner (stream);
  mu_off_t fs_offset = fs->offset;

  if (offset < fs_offset)
    return mu_stream_read (fs->cache, optr, osize, offset, pnbytes);
  else if (offset > fs_offset)
    {
      int status = 0;
      size_t n, left = offset - fs_offset + 1;
      char *buf = malloc (left);
      if (!buf)
	return ENOMEM;
      while (left > 0
	     && (status = mu_stream_read (stream, buf, left, fs_offset, &n)) == 0
	     && n > 0)
	{
	  size_t k;
	  status = mu_stream_write (fs->cache, buf, n, fs_offset, &k);
	  if (status)
	    break;
	  if (k != n)
	    {
	      status = EIO;
	      break;
	    }
	  
	  fs_offset += n;
	  left -= n;
	}
      free (buf);
      if (status)
	return status;
    }
  
  if (feof (fs->file))
    nbytes = 0;
  else
    {
      status = _file_read (stream, optr, osize, fs_offset, &nbytes);
      if (status == 0 && nbytes)
	{
	  size_t k;

	  status = mu_stream_write (fs->cache, optr, nbytes, fs_offset, &k);
	  if (status)
	    return status;
	  if (k != nbytes)
	    return EIO;
	}
    }
  if (pnbytes)
    *pnbytes = nbytes;
  return status;
}

static int
_stdin_file_readline (mu_stream_t stream, char *optr, size_t osize,
		      mu_off_t offset, size_t *pnbytes)
{
  int status;
  size_t nbytes;
  struct _file_stream *fs = mu_stream_get_owner (stream);
  mu_off_t fs_offset = fs->offset;
  
  if (offset < fs->offset)
    return mu_stream_readline (fs->cache, optr, osize, offset, pnbytes);
  else if (offset > fs->offset)
    return ESPIPE;

  fs_offset = fs->offset;
  status = _file_readline (stream, optr, osize, fs_offset, &nbytes);
  if (status == 0)
    {
      size_t k;

      status = mu_stream_write (fs->cache, optr, nbytes, fs_offset, &k);
      if (status)
	return status;
      if (k != nbytes)
	return EIO;
    }
  if (pnbytes)
    *pnbytes = nbytes;
  return status;
}

/* Used only if stream->cache is not NULL */ 
static int
_stdin_file_size (mu_stream_t stream, mu_off_t *psize)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);

  if (!fs->size_computed)
    {
      char buf[512];
      mu_off_t fs_offset = fs->offset;
      size_t n;
      int status;
      
      /* Fill in the cache */ 
      while ((status = mu_stream_read (stream, buf, sizeof (buf),
				       fs_offset, &n)) == 0
	     && n > 0)
	fs_offset += n;
      fs->size = fs_offset;
      fs->size_computed = 1;
    }
  *psize = fs->size;
  return 0;
}

static int
_stdout_file_write (mu_stream_t stream, const char *iptr, size_t isize,
		    mu_off_t offset, size_t *nbytes)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  return _file_write (stream, iptr, isize, fs->offset, nbytes);
}

static int
_file_truncate (mu_stream_t stream, mu_off_t len)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  if (fs->file && ftruncate (fileno(fs->file), len) != 0)
    return errno;
  return 0;
}

static int
_file_size (mu_stream_t stream, mu_off_t *psize)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  struct stat stbuf;
  if (!fs->file)
    {
      if (psize)
	*psize = 0;
      return 0;
    }
  fflush (fs->file);
  if (fstat(fileno(fs->file), &stbuf) == -1)
    return errno;
  if (psize)
    *psize = stbuf.st_size;
  return 0;
}

static int
_file_flush (mu_stream_t stream)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  if (fs->file)
    return fflush (fs->file);
  return 0;
}

int
_file_wait (mu_stream_t stream, int *pflags, struct timeval *tvp)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);

  if (!fs->file)
    return EINVAL;
  return mu_fd_wait (fileno (fs->file), pflags, tvp);
}

static int
_file_get_transport2 (mu_stream_t stream,
		      mu_transport_t *pin, mu_transport_t *pout)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  int status = 0;

  if (pin)
    {
      if (fs->file)
	*pin = (mu_transport_t) fs->file;
      else
	status = EINVAL;
    }
  if (pout)
    *pout = NULL;
  return status;
}

static int
_file_close (mu_stream_t stream)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  int err = 0;

  if (!stream)
    return EINVAL;

  if (fs->file)
    {
      int flags = 0;

      mu_stream_get_flags (stream, &flags);

      if ((flags & MU_STREAM_NO_CLOSE) == 0)
	{
	  if (fclose (fs->file) != 0)
	    err = errno;
	}
      
      fs->file = NULL;
    }
  return err;
}

static int
_temp_file_open (mu_stream_t stream)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  int fd;

  fd = mu_tempfile (fs->filename, NULL);
  if (fd == -1)
    return errno;
  fs->file = fdopen (fd, "r+b");
  if (fs->file == NULL)
    return errno;

  return 0;
}

static int
_file_open (mu_stream_t stream)
{
  struct _file_stream *fs = mu_stream_get_owner (stream);
  int flg;
  int fd;
  const char *mode;
  char* filename = 0;
  int flags = 0;
  
  if (!fs || !fs->filename)
    return EINVAL;
  
  filename = fs->filename;

  if (fs->file)
    {
      fclose (fs->file);
      fs->file = NULL;
    }

  mu_stream_get_flags (stream, &flags);

  /* Map the flags to the system equivalent.  */
  if (flags & MU_STREAM_WRITE && flags & MU_STREAM_READ)
    return EINVAL;
  else if (flags & (MU_STREAM_WRITE|MU_STREAM_APPEND))
    flg = O_WRONLY;
  else if (flags & MU_STREAM_RDWR)
    flg = O_RDWR;
  else /* default */
    flg = O_RDONLY;

  /* Local folders should not block it is local disk ???
     We simply ignore the O_NONBLOCK flag
     But take care of the APPEND.  */
  if (flags & MU_STREAM_APPEND)
    flg |= O_APPEND;

  /* Handle CREAT with care, not to follow symlinks.  */
  if (flags & MU_STREAM_CREAT)
    {
      /* First see if the file already exists.  */
      fd = open (filename, flg);
      if (fd == -1)
	{
	  /* Oops bail out.  */
	  if (errno != ENOENT)
	    return errno;
	  /* Race condition here when creating the file ??.  */
	  fd = open (filename, flg|O_CREAT|O_EXCL,
		     0600 | mu_stream_flags_to_mode (flags, 0));
	  if (fd < 0)
	    return errno;
	}
    }
  else
    {
      fd = open (filename, flg);
      if (fd < 0)
        return errno;
    }

  /* We have to make sure that We did not open
     a symlink. From Casper D. in bugtraq.  */
  if (flg & (MU_STREAM_CREAT | MU_STREAM_RDWR
	     | MU_STREAM_WRITE | MU_STREAM_APPEND))
    {
      struct stat fdbuf, filebuf;

      /* The next two stats should never fail.  */
      if (fstat (fd, &fdbuf) == -1)
	return errno;
      if (lstat (filename, &filebuf) == -1)
	return errno;

      /* Now check that: file and fd reference the same file,
	 file only has one link, file is plain file.  */
      if (!(flags & MU_STREAM_ALLOW_LINKS)
	  && (fdbuf.st_dev != filebuf.st_dev
	      || fdbuf.st_ino != filebuf.st_ino
	      || fdbuf.st_nlink != 1
	      || filebuf.st_nlink != 1
	      || (fdbuf.st_mode & S_IFMT) != S_IFREG))
	{
	  mu_error (_("%s must be a plain file with one link"), filename);
	  close (fd);
	  return EINVAL;
	}
    }
  /* We use FILE * object.  */
  if (flags & MU_STREAM_APPEND)
    mode = "a";
  else if (flags & MU_STREAM_RDWR)
    mode = "r+b";
  else if (flags & MU_STREAM_WRITE)
    mode = "wb";
  else /* Default readonly.  */
    mode = "rb";

  fs->file = fdopen (fd, mode);
  if (fs->file == NULL)
    return errno;

  return 0;
}

int
_file_strerror (mu_stream_t unused, const char **pstr)
{
  *pstr = strerror (errno);
  return 0;
}

int
mu_file_stream_create (mu_stream_t *stream, const char* filename, int flags)
{
  struct _file_stream *fs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  fs = calloc (1, sizeof (struct _file_stream));
  if (fs == NULL)
    return ENOMEM;

  if ((fs->filename = strdup(filename)) == NULL)
    {
      free (fs);
      return ENOMEM;
    }

  ret = mu_stream_create (stream, flags|MU_STREAM_NO_CHECK, fs);
  if (ret != 0)
    {
      free (fs->filename);
      free (fs);
      return ret;
    }

  mu_stream_set_open (*stream, _file_open, fs);
  mu_stream_set_close (*stream, _file_close, fs);
  mu_stream_set_get_transport2 (*stream, _file_get_transport2, fs);
  mu_stream_set_read (*stream, _file_read, fs);
  mu_stream_set_readline (*stream, _file_readline, fs);
  mu_stream_set_write (*stream, _file_write, fs);
  mu_stream_set_truncate (*stream, _file_truncate, fs);
  mu_stream_set_size (*stream, _file_size, fs);
  mu_stream_set_flush (*stream, _file_flush, fs);
  mu_stream_set_destroy (*stream, _file_destroy, fs);
  mu_stream_set_strerror (*stream, _file_strerror, fs);
  mu_stream_set_wait (*stream, _file_wait, fs);
  
  return 0;
}

int
mu_temp_file_stream_create (mu_stream_t *stream, const char *dir)
{
  struct _file_stream *fs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  fs = calloc (1, sizeof (struct _file_stream));
  if (fs == NULL)
    return ENOMEM;
  fs->tempfile = 1;

  if (!dir)
    fs->filename = NULL;
  else if ((fs->filename = strdup (dir)) == NULL)
    {
      free (fs);
      return ENOMEM;
    }
  
  ret = mu_stream_create (stream,
			  MU_STREAM_RDWR|MU_STREAM_CREAT|MU_STREAM_NO_CHECK,
			  fs);
  if (ret != 0)
    {
      free (fs);
      return ret;
    }

  mu_stream_set_open (*stream, _temp_file_open, fs);
  mu_stream_set_close (*stream, _file_close, fs);
  mu_stream_set_get_transport2 (*stream, _file_get_transport2, fs);
  mu_stream_set_read (*stream, _file_read, fs);
  mu_stream_set_readline (*stream, _file_readline, fs);
  mu_stream_set_write (*stream, _file_write, fs);
  mu_stream_set_truncate (*stream, _file_truncate, fs);
  mu_stream_set_size (*stream, _file_size, fs);
  mu_stream_set_flush (*stream, _file_flush, fs);
  mu_stream_set_destroy (*stream, _file_destroy, fs);
  mu_stream_set_strerror (*stream, _file_strerror, fs);
  mu_stream_set_wait (*stream, _file_wait, fs);
  
  return 0;
}

int
mu_stdio_stream_create (mu_stream_t *stream, FILE *file, int flags)
{
  struct _file_stream *fs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (file == NULL)
    return EINVAL;

  fs = calloc (1, sizeof (struct _file_stream));
  if (fs == NULL)
    return ENOMEM;

  fs->file = file;

  ret = mu_stream_create (stream, flags|MU_STREAM_NO_CHECK, fs);
  if (ret != 0)
    {
      free (fs);
      return ret;
    }

  /* Check if we need to enable caching */

  if ((flags & MU_STREAM_SEEKABLE) && lseek (fileno (file), 0, 0))
    {
      if ((ret = mu_memory_stream_create (&fs->cache, 0, MU_STREAM_RDWR))
	  || (ret = mu_stream_open (fs->cache)))
	{
	  mu_stream_destroy (stream, fs);
	  free (fs);
	  return ret;
	}
      mu_stream_set_read (*stream, _stdin_file_read, fs);
      mu_stream_set_readline (*stream, _stdin_file_readline, fs);
      mu_stream_set_write (*stream, _stdout_file_write, fs);
      mu_stream_set_size (*stream, _stdin_file_size, fs);
    }
  else
    {
      mu_stream_set_read (*stream, _file_read, fs);
      mu_stream_set_readline (*stream, _file_readline, fs);
      mu_stream_set_write (*stream, _file_write, fs);
    }
  
  /* We don't need to open the FILE, just return success. */

  mu_stream_set_open (*stream, NULL, fs);
  mu_stream_set_close (*stream, _file_close, fs);
  mu_stream_set_get_transport2 (*stream, _file_get_transport2, fs);
  mu_stream_set_flush (*stream, _file_flush, fs);
  mu_stream_set_destroy (*stream, _file_destroy, fs);
  mu_stream_set_wait (*stream, _file_wait, fs);
  
  return 0;
}


struct _prog_stream
{
  pid_t pid;
  int status;
  pid_t writer_pid;
  int argc;
  char **argv;
  mu_stream_t in, out;

  mu_stream_t input;
};

static mu_list_t prog_stream_list;

static int
_prog_stream_register (struct _prog_stream *stream)
{
  if (!prog_stream_list)
    {
      int rc = mu_list_create (&prog_stream_list);
      if (rc)
	return rc;
    }
  return mu_list_append (prog_stream_list, stream);
}

static void
_prog_stream_unregister (struct _prog_stream *stream)
{
  mu_list_remove (prog_stream_list, stream);
}

#if defined (HAVE_SYSCONF) && defined (_SC_OPEN_MAX)
# define getmaxfd() sysconf (_SC_OPEN_MAX)
#elif defined (HAVE_GETDTABLESIZE)
# define getmaxfd() getdtablesize ()
#else
# define getmaxfd() 64
#endif

#define REDIRECT_STDIN_P(f) ((f) & (MU_STREAM_WRITE|MU_STREAM_RDWR))
#define REDIRECT_STDOUT_P(f) ((f) & (MU_STREAM_READ|MU_STREAM_RDWR))

static int
start_program_filter (pid_t *pid, int *p, int argc, char **argv,
		      char *errfile, int flags)
{
  int rightp[2], leftp[2];
  int i;
  int rc = 0;
  
  if (REDIRECT_STDIN_P (flags))
    pipe (leftp);
  if (REDIRECT_STDOUT_P (flags))
    pipe (rightp);
  
  switch (*pid = fork ())
    {
      /* The child branch.  */
    case 0:
      /* attach the pipes */

      /* Right-end */
      if (REDIRECT_STDOUT_P (flags))
	{
	  if (rightp[1] != 1)
	    {
	      close (1);
	      dup2 (rightp[1], 1);
	    }
	  close (rightp[0]);
	}

      /* Left-end */
      if (REDIRECT_STDIN_P (flags))
	{
	  if (leftp[0] != 0)
	    {
	      close (0);
	      dup2 (leftp[0], 0);
	    }
	  close (leftp[1]);
	}
      
      /* Error output */
      if (errfile)
	{
	  i = open (errfile, O_CREAT|O_WRONLY|O_APPEND, 0644);
	  if (i > 0 && i != 2)
	    {
	      dup2 (i, 2);
	      close (i);
	    }
	}
      /* Close unneded descripitors */
      for (i = getmaxfd (); i > 2; i--)
	close (i);

      syslog (LOG_ERR|LOG_USER, "run %s %s",
	      argv[0], argv[1]);
      /*FIXME: Switch to other uid/gid if desired */
      execvp (argv[0], argv);
		
      /* Report error via syslog */
      syslog (LOG_ERR|LOG_USER, "can't run %s (ruid=%d, euid=%d): %m",
	      argv[0], getuid (), geteuid ());
      exit (127);
      /********************/

      /* Parent branches: */
    case -1:
      /* Fork has failed */
      /* Restore things */
      rc = errno;
      if (REDIRECT_STDOUT_P (flags))
	{
	  close (rightp[0]);
	  close (rightp[1]);
	}
      if (REDIRECT_STDIN_P (flags))
	{
	  close (leftp[0]);
	  close (leftp[1]);
	}
      break;
		
    default:
      if (REDIRECT_STDOUT_P (flags))
	{
	  p[0] = rightp[0];
	  close (rightp[1]);
	}
      else
	p[0] = -1;

      if (REDIRECT_STDIN_P (flags))
	{
	  p[1] = leftp[1];
	  close (leftp[0]);
	}
      else
	p[1] = -1;
    }
  return rc;
}

static void
_prog_wait (pid_t pid, int *pstatus)
{
  if (pid > 0)
    {
      pid_t t;
      do
	t = waitpid (pid, pstatus, 0);
      while (t == -1 && errno == EINTR);
    }
}

static void
_prog_destroy (mu_stream_t stream)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  int status;
    
  mu_argcv_free (fs->argc, fs->argv);
  if (fs->in)
    mu_stream_destroy (&fs->in, mu_stream_get_owner (fs->in));
  if (fs->out)
    mu_stream_destroy (&fs->out, mu_stream_get_owner (fs->out));
  
  _prog_wait (fs->pid, &fs->status);
  fs->pid = -1;
  _prog_wait (fs->writer_pid, &status);
  fs->writer_pid = -1;
  
  _prog_stream_unregister (fs);
}

static int
_prog_close (mu_stream_t stream)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  int status;
  
  if (!stream)
    return EINVAL;
  
  if (fs->pid <= 0)
    return 0;

  mu_stream_close (fs->out);
  mu_stream_destroy (&fs->out, mu_stream_get_owner (fs->out));

  _prog_wait (fs->pid, &fs->status);
  fs->pid = -1;
  _prog_wait (fs->writer_pid, &status);
  fs->writer_pid = -1;
  
  mu_stream_close (fs->in);
  mu_stream_destroy (&fs->in, mu_stream_get_owner (fs->in));

  if (WIFEXITED (fs->status))
    {
      if (WEXITSTATUS (fs->status) == 0)
	return 0;
      else if (WEXITSTATUS (fs->status) == 127)
	return MU_ERR_PROCESS_NOEXEC;
      else
	return MU_ERR_PROCESS_EXITED;
    }
  else if (WIFSIGNALED (fs->status))
    return MU_ERR_PROCESS_SIGNALED;
  return MU_ERR_PROCESS_UNKNOWN_FAILURE;
}

static int
feed_input (struct _prog_stream *fs)
{
  pid_t pid;
  size_t size;
  char buffer[128];
  int rc = 0;

  pid = fork ();
  switch (pid)
    {
    default:
      /* Master */
      fs->writer_pid = pid;
      mu_stream_close (fs->out);
      mu_stream_destroy (&fs->out, mu_stream_get_owner (fs->out));
      break;
      
    case 0:
      /* Child */
      while (mu_stream_sequential_read (fs->input, buffer, sizeof (buffer),
				     &size) == 0
	     && size > 0)
	mu_stream_sequential_write (fs->out, buffer, size);
      mu_stream_close (fs->out);
      exit (0);
      
    case -1:
      rc = errno;
    }

  return rc;
}
  
static int
_prog_open (mu_stream_t stream)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  int rc;
  int pfd[2];
  int flags;
  int seekable_flag;
  
  if (!fs || fs->argc == 0)
    return EINVAL;

  if (fs->pid)
    {
      _prog_close (stream);
    }

  mu_stream_get_flags (stream, &flags);
  seekable_flag = (flags & MU_STREAM_SEEKABLE);
  
  rc = start_program_filter (&fs->pid, pfd, fs->argc, fs->argv, NULL, flags);
  if (rc)
    return rc;

  if (REDIRECT_STDOUT_P (flags))
    {
      FILE *fp = fdopen (pfd[0], "r");
      setvbuf (fp, NULL, _IONBF, 0);
      rc = mu_stdio_stream_create (&fs->in, fp, MU_STREAM_READ|seekable_flag);
      if (rc)
	{
	  _prog_close (stream);
	  return rc;
	}
      rc = mu_stream_open (fs->in);
      if (rc)
	{
	  _prog_close (stream);
	  return rc;
	}
    }
  
  if (REDIRECT_STDIN_P (flags))
    {
      FILE *fp = fdopen (pfd[1], "w");
      setvbuf (fp, NULL, _IONBF, 0);
      rc = mu_stdio_stream_create (&fs->out, fp, MU_STREAM_WRITE|seekable_flag);
      if (rc)
	{
	  _prog_close (stream);
	  return rc;
	}
      rc = mu_stream_open (fs->out);
      if (rc)
	{
	  _prog_close (stream);
	  return rc;
	}
    }

  _prog_stream_register (fs);
  if (fs->input)
    return feed_input (fs);
  return 0;
}

static int
_prog_read (mu_stream_t stream, char *optr, size_t osize,
	    mu_off_t offset, size_t *pnbytes)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  return mu_stream_read (fs->in, optr, osize, offset, pnbytes);
}

static int
_prog_readline (mu_stream_t stream, char *optr, size_t osize,
		mu_off_t offset, size_t *pnbytes)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  return mu_stream_readline (fs->in, optr, osize, offset, pnbytes);
}

static int
_prog_write (mu_stream_t stream, const char *iptr, size_t isize,
	     mu_off_t offset, size_t *pnbytes)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  return mu_stream_write (fs->out, iptr, isize, offset, pnbytes);
}

static int
_prog_flush (mu_stream_t stream)
{
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  mu_stream_flush (fs->in);
  mu_stream_flush (fs->out);
  return 0;
}

static int
_prog_get_transport2 (mu_stream_t stream, mu_transport_t *pin, mu_transport_t *pout)
{
  int rc;
  struct _prog_stream *fs = mu_stream_get_owner (stream);
  
  if ((rc = mu_stream_get_transport (fs->in, pin)) != 0)
    return rc;
  return mu_stream_get_transport (fs->out, pout);
}

int
_prog_stream_create (struct _prog_stream **pfs,
		     mu_stream_t *stream, const char *progname, int flags)
{
  struct _prog_stream *fs;
  int ret;

  if (stream == NULL)
    return MU_ERR_OUT_PTR_NULL;

  if (progname == NULL || (flags & MU_STREAM_NO_CLOSE))
    return EINVAL;

  if ((flags & (MU_STREAM_READ|MU_STREAM_WRITE)) ==
      (MU_STREAM_READ|MU_STREAM_WRITE))
    {
      flags &= ~(MU_STREAM_READ|MU_STREAM_WRITE);
      flags |= MU_STREAM_RDWR;
    }
  
  fs = calloc (1, sizeof (*fs));
  if (fs == NULL)
    return ENOMEM;
  if (mu_argcv_get (progname, "", "#", &fs->argc, &fs->argv))
    {
      mu_argcv_free (fs->argc, fs->argv);
      free (fs);
      return ENOMEM;
    }

  ret = mu_stream_create (stream, flags|MU_STREAM_NO_CHECK, fs);
  if (ret != 0)
    {
      mu_argcv_free (fs->argc, fs->argv);
      free (fs);
      return ret;
    }

  mu_stream_set_read (*stream, _prog_read, fs);
  mu_stream_set_readline (*stream, _prog_readline, fs);
  mu_stream_set_write (*stream, _prog_write, fs);
  
  /* We don't need to open the FILE, just return success. */

  mu_stream_set_open (*stream, _prog_open, fs);
  mu_stream_set_close (*stream, _prog_close, fs);
  mu_stream_set_get_transport2 (*stream, _prog_get_transport2, fs);
  mu_stream_set_flush (*stream, _prog_flush, fs);
  mu_stream_set_destroy (*stream, _prog_destroy, fs);

  if (pfs)
    *pfs = fs;
  return 0;
}

int
mu_prog_stream_create (mu_stream_t *stream, const char *progname, int flags)
{
  return _prog_stream_create (NULL, stream, progname, flags);
}

int
mu_filter_prog_stream_create (mu_stream_t *stream, const char *progname,
			      mu_stream_t input)
{
  struct _prog_stream *fs;
  int rc = _prog_stream_create (&fs, stream, progname, MU_STREAM_RDWR);
  if (rc)
    return rc;
  fs->input = input;

  return 0;
}

