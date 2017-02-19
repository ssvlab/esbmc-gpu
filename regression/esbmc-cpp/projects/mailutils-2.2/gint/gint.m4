# gint.m4 serial 1
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

dnl The option stuff below is based on the similar code from Automake

# _GINT_MANGLE_OPTION(NAME)
# -------------------------
# Convert NAME to a valid m4 identifier, by replacing invalid characters
# with underscores, and prepend the _GINT_OPTION_ suffix to it.
AC_DEFUN([_GINT_MANGLE_OPTION],
[[_GINT_OPTION_]m4_bpatsubst($1, [[^a-zA-Z0-9_]], [_])])

# _GINT_SET_OPTION(NAME)
# ----------------------
# Set option NAME.  If NAME begins with a digit, treat it as a requested
# Guile version number, and define _GINT_GUILE_VERSION to that number.
# Otherwise, define the option using _GINT_MANGLE_OPTION.
AC_DEFUN([_GINT_SET_OPTION],
[m4_if(m4_bpatsubst($1,^[[0-9]].*,[]),,[m4_define([_GINT_GUILE_VERSION],[$1])],[m4_define(_GINT_MANGLE_OPTION([$1]), 1)])])

# _GINT_IF_OPTION_SET(NAME,IF-SET,IF-NOT-SET)
# -------------------------------------------
# Check if option NAME is set.
AC_DEFUN([_GINT_IF_OPTION_SET],
[m4_ifset(_GINT_MANGLE_OPTION([$1]),[$2],[$3])])

# _GINT_OPTION_SWITCH(NAME1,IF-SET1,[NAME2,IF-SET2,[...]],[IF-NOT-SET])
# ------------------------------------------------------------------------
# If NAME1 is set, run IF-SET1.  Otherwise, if NAME2 is set, run IF-SET2.
# Continue the process for all name-if-set pairs within [...].  If none
# of the options is set, run IF-NOT-SET.
AC_DEFUN([_GINT_OPTION_SWITCH],
[m4_if([$4],,[_GINT_IF_OPTION_SET($@)],dnl
[$3],,[_GINT_IF_OPTION_SET($@)],dnl
[_GINT_IF_OPTION_SET([$1],[$2],[_GINT_OPTION_SWITCH(m4_shift(m4_shift($@)))])])])

# _GINT_SET_OPTIONS(OPTIONS)
# ----------------------------------
# OPTIONS is a space-separated list of Gint options.
AC_DEFUN([_GINT_SET_OPTIONS],
[m4_foreach_w([_GINT_Option], [$1], [_GINT_SET_OPTION(_GINT_Option)])])

AC_SUBST(GINT_INCLUDES)
AC_SUBST(GINT_LDADD)

dnl GINT_INIT([DIR], [OPTIONS], [IF-FOUND], [IF-NOT-FOUND])
dnl -------------------------------------------------------
dnl DIR           Gint submodule directory (defaults to 'gint')
dnl OPTIONS       A whitespace-separated list of options. Currently recognized
dnl               options are: 'inc', 'std-site-dir','snarf-doc-filter',
dnl               'nodoc', and version number.
dnl IF-FOUND      What to do if Guile is present.
dnl IF-NOT-FOUND  What to do otherwise.
dnl
AC_DEFUN([GINT_INIT],[
  AM_PROG_LEX
  _GINT_SET_OPTIONS([$2])
  AC_SUBST([GINT_MODULE_DIR],[m4_if([$1],,[gint],[$1])])
  AM_CONDITIONAL([GINT_COND_INC],[_GINT_IF_OPTION_SET([inc],[true],[false])])
  AM_CONDITIONAL([GINT_COND_DOC],[_GINT_IF_OPTION_SET([nodoc],[false],[true])])
  AM_CONDITIONAL([GINT_COND_SNARF_DOC_FILTER],dnl
                 [_GINT_IF_OPTION_SET([snarf-doc-filter],[true],[false])])
  GINT_CHECK_GUILE(m4_ifdef([_GINT_GUILE_VERSION],_GINT_GUILE_VERSION),[$3],[$4])
])

