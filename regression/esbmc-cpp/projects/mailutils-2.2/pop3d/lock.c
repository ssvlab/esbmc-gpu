/* GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
   Free Software Foundation, Inc.

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

#include "pop3d.h"

int
pop3d_lock ()
{
  mu_url_t url = NULL;
  mu_locker_t lock = NULL;
  const char *name;
  int status;

  mu_mailbox_get_url (mbox, &url);
  name = mu_url_to_string (url);
  mu_mailbox_get_locker (mbox, &lock);
  mu_locker_mod_flags (lock, MU_LOCKER_PID, mu_locker_set_bit);
  if ((status = mu_locker_lock (lock)))
    {
      mu_diag_output (MU_DIAG_NOTICE, _("locking mailbox `%s' failed: %s"),
	      name ? name : "?", mu_strerror(status));
      return ERR_MBOX_LOCK;
    }
  return 0;
}

int
pop3d_touchlock ()
{
  mu_locker_t lock = NULL;
  mu_mailbox_get_locker (mbox, &lock);
  mu_locker_touchlock (lock);
  return 0;
}

int
pop3d_unlock ()
{
  mu_locker_t lock = NULL;
  mu_mailbox_get_locker (mbox, &lock);
  mu_locker_unlock (lock);
  return 0;
}
