## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2002, 2003, 2004, 2007, 2008, 2009, 2010 Free
##   Software Foundation, Inc.
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
##   Foundation, Inc. 51 Franklin Street, Fifth Floor, Boston, MA
##   02110-1301 USA

SUBDIRS = config cpp python scheme

noinst_PROGRAMS = \
 aclck\
 addr\
 argcv\
 base64\
 decode2047\
 echosrv\
 encode2047\
 header\
 http\
 iconv\
 listop\
 lsf\
 mailcap\
 mimetest\
 msg-send\
 mta\
 muauth\
 muemail\
 murun\
 nntpclient\
 pop3client \
 sfrom\
 url-parse

## NOTE: Numaddr must be an installable target, otherwise libtool
## will not create a shared library and `make check' will fail in
## sieve. Pity.
## -Sergey
moddir=@MU_SIEVE_MODDIR@
mod_LTLIBRARIES = numaddr.la

INCLUDES = @MU_LIB_COMMON_INCLUDES@ 

LDADD = \
 ../lib/libmuaux.la\
 ${MU_LIB_MAILUTILS}\
 @MU_COMMON_LIBRARIES@

numaddr_la_SOURCES = numaddr.c
numaddr_la_LIBADD = ${MU_LIB_SIEVE}
numaddr_la_LDFLAGS = -module -avoid-version -no-undefined -rpath '$(moddir)'

mimetest_LDADD =\
 ../lib/libmuaux.la\
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}

msg_send_LDADD =\
 ../lib/libmuaux.la\
 ${MU_LIB_MAILER}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}

mta_CPPFLAGS=-I${top_srcdir}/lib -I${top_builddir}/lib
mta_LDADD =\
 ../lib/libmuaux.la\
 ${MU_LIB_MBOX}\
 ${MU_LIB_MAILER}\
 ${MU_LIB_MAILUTILS}

sfrom_LDADD =\
 ../lib/libmuaux.la\
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}

pop3client_LDADD = \
 ../lib/libmuaux.la\
 ${MU_LIB_POP}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}\
 @READLINE_LIBS@

nntpclient_LDADD = \
 ../lib/libmuaux.la\
 ${MU_LIB_NNTP}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}\
 @READLINE_LIBS@

lsf_LDADD = \
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@ \
 ${MU_LIB_MAILUTILS}

muauth_CPPFLAGS = @MU_APP_COMMON_INCLUDES@
muauth_LDADD = \
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@ \
 ${MU_LIB_MAILUTILS}

muemail_CPPFLAGS = @MU_APP_COMMON_INCLUDES@
muemail_LDADD = \
 ${MU_APP_LIBRARIES} \
 ${MU_LIB_MAILUTILS}

