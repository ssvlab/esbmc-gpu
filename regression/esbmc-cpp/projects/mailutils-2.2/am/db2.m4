dnl This file is part of GNU mailutils.
dnl Copyright (C) 2003, 2007, 2010 Free Software Foundation, Inc.
dnl
dnl GNU Mailutils is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3, or (at your option)
dnl any later version.
dnl 
dnl GNU Mailutils is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl 
dnl You should have received a copy of the GNU General Public License along
dnl with GNU Mailutils; if not, write to the Free Software Foundation,
dnl Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
dnl

AH_TEMPLATE(BDB2_CURSOR_LASTARG,
            [Last argument to the cursor member of Berkeley 2 DB structure])

dnl The cursor member of DB structure used to take three arguments in older
dnl implementations of Berkeley DB. Newer versions (>= 4.0) declare
dnl it as taking four arguments.
dnl This macro checks which of the variants we have.
AC_DEFUN([MU_DB2_CURSOR],
 [AC_CACHE_CHECK([whether db->cursor takes 4 arguments],
                 [mu_cv_bdb2_cursor_four_args],
  [AC_TRY_COMPILE([#include <db.h>],
                  [
DB *db;
db->cursor(NULL, NULL, NULL, 0)
                  ],
                  [mu_cv_bdb2_cursor_four_args=yes],
                  [mu_cv_bdb2_cursor_four_args=no])])
 if test $mu_cv_bdb2_cursor_four_args = yes; then
   AC_DEFINE(BDB2_CURSOR_LASTARG,[,0])
 else
   AC_DEFINE(BDB2_CURSOR_LASTARG,[])
 fi])


