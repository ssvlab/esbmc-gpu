## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2008, 2009, 2010
##   Free Software Foundation, Inc.
##
##   GNU Mailutils is free software; you can redistribute it and/or
##   modify it under the terms of the GNU General Public License as
##   published by the Free Software Foundation; either version 3, or (at
##   your option) any later version.
##
##   This program is distributed in the hope that it will be useful, but
##   WITHOUT ANY WARRANTY; without even the implied warranty of
##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##   General Public License for more details.
##
##   You should have received a copy of the GNU General Public License
##   along with this program; if not, write to the Free Software
##   Foundation, Inc.  51 Franklin Street, Fifth Floor, Boston, MA
##   02110-1301 USA

include gnulib.mk

INCLUDES = @MU_LIB_COMMON_INCLUDES@ 

libmuaux_la_SOURCES += \
 daemon.c\
 mailcap.c\
 mu_dbm.c\
 signal.c\
 strexit.c\
 tcpwrap.c\
 userprivs.c\
 mu_umaxtostr.c\
 mu_umaxtostr.h

noinst_HEADERS +=\
 mailcap.h\
 muaux.h\
 mu_dbm.h\
 mu_asprintf.h\
 tcpwrap.h

EXTRA_DIST += utmp.c
gl_LIBOBJS += @LIBOBJS@
gl_LTLIBOBJS += @LTLIBOBJS@


