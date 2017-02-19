dnl This file is part of GNU mailutils.
dnl Copyright (C) 2003, 2007, 2009, 2010 Free Software Foundation, Inc.
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
AC_DEFUN([MU_CHECK_TLS],
[
 if test "x$WITH_GNUTLS" = x; then
   cached=""
   AC_ARG_WITH([gnutls],
               AC_HELP_STRING([--without-gnutls],
                              [do not use GNU TLS library]),
               [WITH_GNUTLS=$withval],
               [WITH_GNUTLS=yes])

   if test "$WITH_GNUTLS" != "no"; then
     AC_CHECK_HEADER(gnutls/gnutls.h,
                     [:],
                     [WITH_GNUTLS=no])
     if test "$WITH_GNUTLS" != "no"; then
       saved_LIBS=$LIBS
       AC_CHECK_LIB(gcrypt, main,
                    [TLS_LIBS="-lgcrypt"],
                    [WITH_GNUTLS=no])
       LIBS="$LIBS $TLS_LIBS"
       AC_CHECK_LIB(gnutls, gnutls_global_init,
                    [TLS_LIBS="-lgnutls $TLS_LIBS"],
                    [WITH_GNUTLS=no])
       LIBS=$saved_LIBS
     fi
   fi
 else
  cached=" (cached) "
 fi
 AC_MSG_CHECKING([whether to use TLS libraries])
 AC_MSG_RESULT(${cached}${WITH_GNUTLS})])
