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

int mu_switch_to_privs (uid_t uid, gid_t gid, mu_list_t retain_groups);
void mu_set_signals (RETSIGTYPE (*handler) (int signo), int *sigv, int sigc);
const char *mu_strexit (int code);

