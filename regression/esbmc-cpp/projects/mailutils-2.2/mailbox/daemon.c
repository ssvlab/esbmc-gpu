/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2005, 2007, 2008, 2009, 2010 Free Software
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

#include <mailutils/daemon.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/nls.h>

static char *pidfile;
static pid_t current_pid;

/* Return 0 if DIR is writable for EUID/EGID.
   Otherwise, return error code. */
static int
ewraccess (const char *dir)
{
  struct stat st;
  if (stat (dir, &st))
    return errno;
  if ((st.st_mode & S_IWOTH)
      || (st.st_gid == getegid () && (st.st_mode & S_IWGRP))
      || (st.st_uid == geteuid () && (st.st_mode & S_IWUSR)))
    return 0;
  else
    return EACCES;
}

/* Return 0 if DIR is writable. If necessary and possible, raise to
   EUID 0, in that case return prior EUID in the memory location pointed to
   by PUID. */
static int
access_dir (const char *dir, uid_t *puid)
{
  int ec = ewraccess (dir);
  if (ec)
    {
      if (ec == EACCES && access (dir, W_OK) == 0)
	{
	  uid_t uid = geteuid ();
	  /* See if we can become root */
	  if (uid && getuid () == 0 && seteuid (0) == 0)
	    {
	      *puid = uid;
	      return 0;
	    }
	}
    }
  return ec;
}

int
mu_daemon_create_pidfile (const char *filename)
{
  char *p;
  int fd;
  uid_t uid = 0;
  int rc;
  
  if (filename[0] != '/')
    return EINVAL;

  if (pidfile)
    free (pidfile);
  pidfile = strdup (filename);
  if (!pidfile)
    return ENOMEM;

  /* Determine the hosting directory name */
  p = strrchr (pidfile, '/');
  if (pidfile == p)
    {
      free (pidfile);
      pidfile = NULL;
      /* Sorry, pidfiles in root dir are not allowed */
      return EINVAL;
    }
  /* Check if we have write access to the directory */
  *p = 0;
  rc = access_dir (pidfile, &uid);
  if (rc)
    {
      /* Nope, clean up and return */
      free (pidfile);
      pidfile = NULL;
      return rc;
    }

  /* Restore directory separator */
  *p = '/';
    
  unlink (pidfile);
  current_pid = getpid ();
  
  if ((fd = open (pidfile, O_WRONLY | O_CREAT | O_TRUNC | O_EXCL, 0644)) != -1)
    {
      FILE *fp = fdopen (fd, "w");
      if (!fp)
	{
	  rc = errno;
	  free (pidfile);
	  close (fd);
	}
      else
	{
	  fprintf (fp, "%lu", (unsigned long) current_pid);
	  fclose (fp);
	  atexit (mu_daemon_remove_pidfile);
	}
    }
  else
    {
      rc = errno;
      free (pidfile);
      pidfile = NULL;
    }

  /* Restore previous EUID value. */
  if (uid)
    seteuid (uid);
  
  return rc;
}

void
mu_daemon_remove_pidfile (void)
{
  if (getpid () == current_pid)
    {
      int rc;
      uid_t uid = 0;

      /* Determine the hosting directory name */
      char *p = strrchr (pidfile, '/');
      if (pidfile == p)
	{
	  /* Should not happen */
	  abort ();
	}
      /* Check if we have write access to the directory */
      *p = 0;
      rc = access_dir (pidfile, &uid);
      *p = '/';
      if (rc == 0)
	{
	  if (unlink (pidfile) && errno != ENOENT)
	    rc = errno;
	  else
	    rc = 0;
	}
      
      if (rc)
	mu_error (_("cannot remove pidfile %s: %s"),
		  pidfile, mu_strerror (rc));

      free (pidfile);
      pidfile = NULL;
    }
}



