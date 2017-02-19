# guile.m4 serial 1
dnl This file is part of Gint
dnl Copyright (C) 2010 Sergey Poznyakoff
dnl
dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3, or (at your option)
dnl any later version.
dnl
dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program.  If not, see <http://www.gnu.org/licenses/>.

m4_define([_gint_eval_version],
[m4_if([$2],,[m4_errprint([Version number without dots: $1
])],[m4_eval([$1] * 1000 + [$2] * 100[]m4_if([$3],,,[ + $3]))])])

m4_define([_gint_mangle_version],[_gint_eval_version(m4_bpatsubst($1,\.,[,]))])

m4_define([_gint_site_dir],[`guile -c '(write (%site-dir)) (newline)'`])

dnl GINT_CHECK_GUILE(minversion, [act-if-found], [act-if-not-found])
dnl                      $1            $2               $3
AC_DEFUN([GINT_CHECK_GUILE],
[
  AC_SUBST(GUILE_INCLUDES)
  AC_SUBST(GUILE_LIBS)
  AC_SUBST(GUILE_VERSION)
  AC_SUBST(GUILE_VERSION_NUMBER)
  
  gint_guile_status=no

  _GINT_OPTION_SWITCH(
    [with-guile],
     [AC_ARG_WITH([guile],
	[AC_HELP_STRING([--with-guile],
	                [compile with Guile support (default)])],
	[gint_enable_guile=$withval],
	[gint_enable_guile=yes])],
    [without-guile],
     [AC_ARG_WITH([guile],
	[AC_HELP_STRING([--with-guile],
	                [compile with Guile support])],
	[gint_enable_guile=$withval],
	[gint_enable_guile=no])],
    [enable-guile],
     [AC_ARG_ENABLE([guile],
	[AC_HELP_STRING([--enable-guile],
	                [enable Guile support (default)])],
	[gint_enable_guile=$enableval],
	[gint_enable_guile=yes])],
    [disable-guile],
     [AC_ARG_ENABLE([guile],
	[AC_HELP_STRING([--enable-guile],
	                [enable Guile support])],
	[gint_enable_guile=$enableval],
	[gint_enable_guile=no])],
    [gint_enable_guile=yes])

  if test "$gint_enable_guile" = yes; then
    AC_PATH_PROG(GUILE_CONFIG, guile-config, no, $PATH)
    if test "$GUILE_CONFIG" = no; then
      m4_if([$3],,[AC_MSG_ERROR(cannot find Guile)], [$3])
    else
      AC_PATH_PROG(GUILE_SNARF, guile-snarf)
      AC_PATH_PROG(GUILE_TOOLS, guile-tools)
  
      GUILE_INCLUDES=`$GUILE_CONFIG compile`
      GUILE_LIBS=`$GUILE_CONFIG link`
      GUILE_VERSION=`($GUILE_CONFIG --version 2>&1; echo '')|sed 's/guile-config [[^0-9]]* \([[0-9]][[0-9.]]*\)$/\1/'`
      VEX=`echo $GUILE_VERSION | sed 's/\./ \\\\* 1000 + /;s/\./ \\\\* 100 + /'`
      GUILE_VERSION_NUMBER=`eval expr "$VEX"`

      gint_guile_status=ok

      m4_if([$1],,,[
        if test $GUILE_VERSION_NUMBER -lt _gint_mangle_version($1); then
          m4_if([$3],,
	        [AC_MSG_ERROR([Guile version too old; required at least ]$1)],
	        [gint_guile_status=badversion])
        fi])

      if test $gint_guile_status = ok; then
        save_LIBS=$LIBS
        save_CFLAGS=$CFLAGS
        LIBS="$LIBS $GUILE_LIBS"
        CFLAGS="$CFLAGS $GUILE_INCLUDES"
        AC_LINK_IFELSE(
         [AC_LANG_PROGRAM([#include <libguile.h>],
                          [scm_shell(0, NULL);])],
	 [],
	 [gint_guile_status=cantlink])
        LIBS=$save_LIBS
        CFLAGS=$save_CFLAGS
      fi  
    fi

    if test $gint_guile_status != ok; then
      GUILE_INCLUDES=
      GUILE_LIBS=
      GUILE_VERSION=
      GUILE_VERSION_NUMBER=
      m4_if([$3],,[AC_MSG_ERROR(required library libguile not found)], [$3])
    else
      save_LIBS=$LIBS
      save_CFLAGS=$CFLAGS
      LIBS="$LIBS $GUILE_LIBS"
      CFLAGS="$CFLAGS $GUILE_INCLUDES"
      AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <libguile.h>]],
                        [SCM_DEVAL_P = 1;
                         SCM_BACKTRACE_P = 1;
                         SCM_RECORD_POSITIONS_P = 1;
                         SCM_RESET_DEBUG_MODE;])],
                        [gint_guile_debug=yes],
                        [gint_guile_debug=no])
      if test $gint_guile_debug = yes; then
        AC_DEFINE_UNQUOTED(GUILE_DEBUG_MACROS, 1,
         [Define to 1 if SCM_DEVAL_P, SCM_BACKTRACE_P, SCM_RECORD_POSITIONS_P and SCM_RESET_DEBUG_MODE are defined])
      fi
      AC_CHECK_TYPES([scm_t_off],[],[],[#include <libguile.h>])
      LIBS=$save_LIBS
      CFLAGS=$save_CFLAGS
    
      AC_SUBST(GUILE_SITE)
      _GINT_IF_OPTION_SET([std-site-dir],
         [GUILE_SITE=_gint_site_dir],
         [AC_ARG_WITH([guile-site-dir],
            [AC_HELP_STRING([--with-guile-site-dir=DIR],
                            [specify directory to install guile modules to])],
            [case $withval in
             /*) GUILE_SITE=$withval;;
             yes) GUILE_SITE=_gint_site_dir;;
             *)  AC_MSG_ERROR([Argument to --with-guile-site-dir must be an absolute directory name]);;
             esac],
            [GUILE_SITE=_gint_site_dir
             pfx=$prefix 
             test "x$pfx" = xNONE && pfx=$ac_default_prefix
             case $GUILE_SITE in
             $pfx/*) ;; # OK
	     *) AC_MSG_WARN([guile site directory $GUILE_SITE lies outside your current prefix ($pfx).])
             GUILE_SITE='$(datadir)/guile/site'
             AC_MSG_WARN([Falling back to ${GUILE_SITE} instead. Use --with-guile-site-dir to force using site directory.])
             ;;
             esac])])
      AC_DEFINE_UNQUOTED(GUILE_VERSION, "$GUILE_VERSION",
                         [Guile version number])
      AC_DEFINE_UNQUOTED(GUILE_VERSION_NUMBER, $GUILE_VERSION_NUMBER,
                         [Guile version number: MAX*10 + MIN])
      m4_if([$2],,,[$2])
    fi
  m4_if([$3],,,[else
    $3])
  fi
])     

