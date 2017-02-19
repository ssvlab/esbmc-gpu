## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2010 Free
##   Software Foundation, Inc.
##
##   GNU Mailutils is free software; you can redistribute it and/or
##   modify it under the terms of the GNU General Public License as
##   published by the Free Software Foundation; either version 3, or (at
##   your option) any later version.
##
##   GNU Mailutils is distributed in the hope that it will be useful, but
##   WITHOUT ANY WARRANTY; without even the implied warranty of
##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##   General Public License for more details.
##
##   You should have received a copy of the GNU General Public License
##   along with GNU Mailutils; if not, write to the Free Software
##   Foundation, Inc.  51 Franklin Street, Fifth Floor, Boston, MA
##   02110-1301 USA

INCLUDES = @MU_APP_COMMON_INCLUDES@ 

sbin_PROGRAMS = pop3d
bin_PROGRAMS = @POPAUTH@
EXTRA_PROGRAMS = popauth
SUBDIRS = testsuite

pop3d_SOURCES =\
 apop.c\
 auth.c\
 bulletin.c\
 capa.c\
 cmd.c\
 dele.c\
 expire.c\
 extra.c\
 list.c\
 lock.c\
 logindelay.c\
 noop.c\
 pop3d.c\
 pop3d.h\
 quit.c\
 retr.c\
 rset.c\
 signal.c\
 stat.c\
 stls.c\
 top.c\
 uidl.c\
 user.c

pop3d_LDADD = \
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_MBOX}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@ \
 ${MU_LIB_MAILUTILS}\
 @MU_COMMON_LIBRARIES@ @TCPWRAP_LIBRARIES@

popauth_SOURCES = popauth.c
popauth_LDADD = ${MU_APP_LIBRARIES} ${MU_LIB_MAILUTILS} @MU_COMMON_LIBRARIES@
pop3d_DEPENDENCIES = \
 @MU_AUTHLIBS_DEPENDENCY@ \
 ../lib/libmuaux.la \
 ${MU_LIB_MBOX}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_MAILUTILS}

