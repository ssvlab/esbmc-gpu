/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2008, 2010 Free Software Foundation, Inc.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>
#include <mailutils/assoc.h>
#include <mailutils/errno.h>
#include <mailutils/error.h>
#include <mailutils/errno.h>
#include <mailutils/nls.h>
#include <mailutils/list.h>
#include <mailutils/iterator.h>
#include <xalloc.h>

/* Switch to the given UID/GID */
int
mu_switch_to_privs (uid_t uid, gid_t gid, mu_list_t retain_groups)
{
  int rc = 0;
  gid_t *emptygidset;
  size_t size = 1, j = 1;
  mu_iterator_t itr;

  if (uid == 0)
    return 0;

  /* Create a list of supplementary groups */
  mu_list_count (retain_groups, &size);
  size++;
  emptygidset = xmalloc (size * sizeof emptygidset[0]);
  emptygidset[0] = gid ? gid : getegid ();

  if (mu_list_get_iterator (retain_groups, &itr) == 0)
    {
      for (mu_iterator_first (itr);
	   !mu_iterator_is_done (itr); mu_iterator_next (itr)) 
	mu_iterator_current (itr,
			     (void **)(emptygidset + j++));
      mu_iterator_destroy (&itr);
    }

  /* Reset group permissions */
  if (geteuid () == 0 && setgroups (j, emptygidset))
    {
      mu_error(_("setgroups(1, %lu) failed: %s"),
	       (unsigned long) emptygidset[0], mu_strerror (errno));
      rc = 1;
    }
  free (emptygidset);
	
  /* Switch to the user's gid. On some OSes the effective gid must
     be reset first */

#if defined(HAVE_SETEGID)
  if ((rc = setegid (gid)) < 0)
    mu_error (_("setegid(%lu) failed: %s"),
	      (unsigned long) gid, mu_strerror (errno));
#elif defined(HAVE_SETREGID)
  if ((rc = setregid (gid, gid)) < 0)
    mu_error (_("setregid(%lu,%lu) failed: %s"),
	      (unsigned long) gid, (unsigned long) gid,
	      mu_strerror (errno));
#elif defined(HAVE_SETRESGID)
  if ((rc = setresgid (gid, gid, gid)) < 0)
    mu_error (_("setresgid(%lu,%lu,%lu) failed: %s"),
	      (unsigned long) gid,
	      (unsigned long) gid,
	      (unsigned long) gid,
	      mu_strerror (errno));
#endif

  if (rc == 0 && gid != 0)
    {
      if ((rc = setgid (gid)) < 0 && getegid () != gid) 
	mu_error (_("setgid(%lu) failed: %s"),
		  (unsigned long) gid, mu_strerror (errno));
      if (rc == 0 && getegid () != gid)
	{
	  mu_error (_("Cannot set effective gid to %lu"),
		    (unsigned long) gid);
	  rc = 1;
	}
    }

  /* Now reset uid */
  if (rc == 0 && uid != 0)
    {
      uid_t euid;

      if (setuid (uid) || geteuid () != uid
	  || (getuid () != uid && (geteuid () == 0 || getuid () == 0)))
	{
#if defined(HAVE_SETREUID)
	  if (geteuid () != uid)
	    {
	      if (setreuid (uid, -1) < 0)
		{
		  mu_error (_("setreuid(%lu,-1) failed: %s"),
			    (unsigned long) uid,
			    mu_strerror (errno));
		  rc = 1;
		}
	      if (setuid (uid) < 0)
		{
		  mu_error (_("second setuid(%lu) failed: %s"),
			    (unsigned long) uid, mu_strerror (errno));
		  rc = 1;
		}
	    } else
#endif
	        {
		  mu_error (_("setuid(%lu) failed: %s"),
			    (unsigned long) uid,
			    mu_strerror (errno));
		  rc = 1;
		}
	}
	
      euid = geteuid ();
      if (uid != 0 && setuid (0) == 0)
	{
	  mu_error (_("seteuid(0) succeeded when it should not"));
	  rc = 1;
	}
      else if (uid != euid && setuid (euid) == 0)
	{
	  mu_error (_("Cannot drop non-root setuid privileges"));
	  rc = 1;
	}
    }
  return rc;
}


