## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 1999, 2000, 2001, 2002, 2003, 2005, 2007, 2008, 2009,
##   2010 Free Software Foundation, Inc.
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

INCLUDES = @MU_APP_COMMON_INCLUDES@ @MU_AUTHINCS@ @SERV_AUTHINCS@ 
SUBDIRS = testsuite

sbin_PROGRAMS =	imap4d

imap4d_SOURCES = \
 append.c\
 authenticate.c\
 bye.c\
 capability.c\
 check.c\
 close.c\
 commands.c\
 copy.c\
 create.c\
 delete.c\
 examine.c\
 expunge.c\
 fetch.c\
 id.c\
 idle.c\
 imap4d.c\
 imap4d.h\
 list.c\
 logout.c\
 login.c\
 lsub.c\
 namespace.c\
 noop.c\
 parsebuf.c\
 preauth.c\
 rename.c\
 search.c\
 select.c\
 signal.c\
 starttls.c\
 status.c\
 store.c\
 subscribe.c\
 sync.c\
 uid.c\
 unsubscribe.c\
 util.c

imap4d_LDADD = \
 @IMAP_AUTHOBJS@\
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_MBOX}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@ \
 ${MU_LIB_MAILUTILS}\
 @SERV_AUTHLIBS@ @MU_COMMON_LIBRARIES@ @TCPWRAP_LIBRARIES@

## This kludge is necessary to correctly establish imap4d -> IMAP_AUTHOBJS
## and imap4d -> MU_AUTHLIBS dependencies. Automake stupidly refuses to 
## include them.
## Same kludge appears in pop3d/Makefile.am and auth/Makefile.am
## Think about better approach         --gray
imap4d_DEPENDENCIES = \
 @IMAP_AUTHOBJS@\
 @MU_AUTHLIBS_DEPENDENCY@\
 ../lib/libmuaux.la\
 ${MU_LIB_MBOX}\
 ${MU_LIB_MAILUTILS}

EXTRA_DIST=auth_gss.c auth_gsasl.c
