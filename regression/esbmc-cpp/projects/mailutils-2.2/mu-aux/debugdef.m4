divert(-1)
# This file is part of Mailutils.
# Copyright (C) 2006, 2007, 2010 Free Software Foundation, Inc.
#
# Initially written by Sergey Poznyakoff for Mailfromd project.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

changecom(/*,*/)

define(`__arglist',`dnl
ifelse($1,$2,x$1,
`x$1, __arglist(incr($1), $2)')')

define(`MKDEBUG',`
`#define __MU_DEBUG'$1`(dbg, lev, fmt, '__arglist(1,$1)) \`
 do \
   { \
     if (mu_debug_line_info) \
       { \
         mu_debug_set_locus (dbg, __FILE__, __LINE__); \
         mu_debug_set_function (dbg, __FUNCTION__); \
       } \
     mu_debug_printf (dbg, lev, fmt, 'dnl
__arglist(1,$1)`); \
   } \
 while (0)

#define MU_DEBUG'$1`(dbg, lev, fmt, '__arglist(1,$1)`) \
 do \
   { \
     if (mu_debug_check_level (dbg, lev)) \
       __MU_DEBUG'$1`(dbg, lev, fmt, '__arglist(1,$1)`); \
   } \
 while(0)
'')

define(`forloop',
       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')
define(`_forloop',
       `$4`'ifelse($1, `$3', ,
              `define(`$1', incr($1))_forloop(`$1', `$2', `$3', `$4')')')

divert(0)dnl
/* -*- buffer-read-only: t -*- vi: set ro:
   THIS FILE IS GENERATED AUTOMATICALLY.  PLEASE DO NOT EDIT.
*/
