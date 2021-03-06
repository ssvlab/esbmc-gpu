# strtok_r.m4 serial 9
dnl Copyright (C) 2002, 2003, 2004, 2007, 2009, 2010 Free Software Foundation,
dnl Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.

AC_DEFUN([gl_FUNC_STRTOK_R],
[
  dnl The strtok_r() declaration in lib/string.in.h uses 'restrict'.
  AC_REQUIRE([AC_C_RESTRICT])

  AC_REQUIRE([gl_HEADER_STRING_H_DEFAULTS])
  AC_REQUIRE([AC_CANONICAL_HOST]) dnl for cross-compiles
  AC_CHECK_FUNCS([strtok_r])
  if test $ac_cv_func_strtok_r = yes; then
    dnl glibc 2.7 has a bug in strtok_r that causes a segmentation fault
    dnl when the second argument to strtok_r is a constant string that has
    dnl exactly one byte and compiling with optimization.  This bug is, for
    dnl example, present in the glibc 2.7-18 package in Debian "lenny".
    dnl See <http://sources.redhat.com/bugzilla/show_bug.cgi?id=5614>.
    AC_CACHE_CHECK([whether strtok_r works], [gl_cv_func_strtok_r_works],
      [AC_RUN_IFELSE(
         [AC_LANG_PROGRAM([[
              #ifndef __OPTIMIZE__
              # define __OPTIMIZE__ 1
              #endif
              #undef __OPTIMIZE_SIZE__
              #undef __NO_INLINE__
              #include <stdlib.h>
              #include <string.h>
            ]],
            [[char delimiters[] = "xxxxxxxx";
              char *save_ptr = (char *) 0xd0d0;
              strtok_r (delimiters, "x", &save_ptr);
              strtok_r (NULL, "x", &save_ptr);
              return 0;
            ]])
         ],
         [gl_cv_func_strtok_r_works=yes],
         [gl_cv_func_strtok_r_works=no],
         [
changequote(,)dnl
          case "$host_os" in
                    # Guess no on glibc systems.
            *-gnu*) gl_cv_func_strtok_r_works="guessing no";;
            *)      gl_cv_func_strtok_r_works="guessing yes";;
          esac
changequote([,])dnl
         ])
      ])
    case "$gl_cv_func_strtok_r_works" in
      *no)
        dnl We could set REPLACE_STRTOK_R=1 and AC_LIBOBJ([strtok_r]) here,
        dnl but it's only the macro version in <bits/string2.h> which is wrong.
        dnl The code compiled into libc is fine.
        UNDEFINE_STRTOK_R=1
        ;;
    esac
  else
    AC_LIBOBJ([strtok_r])
  fi
  AC_CHECK_DECLS_ONCE([strtok_r])
  if test $ac_cv_have_decl_strtok_r = no; then
    HAVE_DECL_STRTOK_R=0
  fi
  gl_PREREQ_STRTOK_R
])

# Prerequisites of lib/strtok_r.c.
AC_DEFUN([gl_PREREQ_STRTOK_R], [
  :
])
