/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2007, 2010 Free Software Foundation,
   Inc.

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
#include <pwd.h>
#include <string.h>
#include <stdlib.h>

/*
  Written by Alain Magloire.
  Simple replacement for fgetpwent(), it is not :
  - thread safe;
  - static buffer was not use since it will limit the size
    of the entry.  But rather memory is allocated and __never__
    release.  The memory will grow if need be.
  - no support for shadow
  - no support for NIS(+)
*/

static char *buffer;
static size_t buflen;
static struct passwd pw;

static char *
parse_line (char *s, char **p)
{
  if (*s)
    {
      char *sep = strchr (s, ':');
      if (sep)
	{
	  *sep++ = '\0';
	  *p = sep;
	}
      else
	*p = s + strlen (s);
    }
  else
    *p = s;
  return s;
}

static struct passwd *
getentry (char *s)
{
  char *p;
  pw.pw_name = parse_line (s, &p);
  s = p;
  pw.pw_passwd = parse_line (s, &p);
  s = p;
  pw.pw_uid = strtoul (parse_line (s, &p), NULL, 10);
  s = p;
  pw.pw_gid = strtoul (parse_line (s, &p), NULL, 10);
  s = p;
  pw.pw_gecos = parse_line (s, &p);
  s = p;
  pw.pw_dir = parse_line (s, &p);
  s = p;
  pw.pw_shell = parse_line (s, &p);
  return &pw;
}

struct passwd *
mu_fgetpwent (FILE *fp)
{
  size_t pos = 0;
  int done = 0;
  struct passwd *pw = NULL;

  /* Allocate buffer if not yet available.  */
  /* This buffer will be never free().  */
  if (buffer == NULL)
    {
      buflen = 1024;
      buffer = malloc (buflen);
      if (buffer == NULL)
	return NULL;
    }

  do
    {
      if (fgets (buffer + pos, buflen, fp) != NULL)
	{
	  /* Need a full line.  */
	  if (buffer[strlen (buffer) - 1] == '\n')
	    {
	      /* reset marker position.  */
	      pos = 0;
	      /* Nuke trailing newline.  */
	      buffer[strlen (buffer) - 1] = '\0';

	      /* Skip comments.  */
	      if (buffer[0] != '#')
		{
		  done = 1;
		  pw = getentry (buffer);
		}
	    }
	  else
	    {
	      /* Line is too long reallocate the buffer.  */
	      char *tmp;
	      pos = strlen (buffer);
	      buflen *= 2;
	      tmp = realloc (buffer, buflen);
	      if (tmp)
		buffer = tmp;
	      else
		done = 1;
	    }
	}
      else
	done = 1;
    } while (!done);

  return pw;

}

#ifdef STANDALONE
int
main ()
{
  FILE *fp = fopen ("/etc/passwd", "r");
  if (fp)
    {
      struct passwd *pwd;
      while ((pwd = fgetpwent (fp)))
        {
          printf ("--------------------------------------\n");
          printf ("name %s\n", pwd->pw_name);
          printf ("passwd %s\n", pwd->pw_passwd);
          printf ("uid %d\n", pwd->pw_uid);
          printf ("gid %d\n", pwd->pw_gid);
          printf ("gecos %s\n", pwd->pw_gecos);
          printf ("dir %s\n", pwd->pw_dir);
          printf ("shell %s\n", pwd->pw_shell);
        }
    }
  return 0;
}

#endif
