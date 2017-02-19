## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2001, 2002, 2006, 2007, 2009, 2010 Free Software
##   Foundation, Inc.
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

INCLUDES = -I. @MU_LIB_COMMON_INCLUDES@

lib_LTLIBRARIES=libmu_scm.la

C_SRCS=\
 mu_address.c\
 mu_body.c\
 mu_dbgport.c\
 mu_guile.c\
 mu_mailbox.c\
 mu_message.c\
 mu_mime.c\
 mu_logger.c\
 mu_port.c\
 mu_scm.c\
 mu_util.c

libmu_scm_la_SOURCES=\
 $(C_SRCS) \
 mu_scm.h

libmu_scm_la_LDFLAGS = -rpath $(libdir) -version-info @VI_CURRENT@:@VI_REVISION@:@VI_AGE@
libmu_scm_la_LIBADD = \
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_MAILER}\
 ${MU_LIB_MAILUTILS}\
 @GUILE_LIBS@

EXTRA_DIST=mailutils.scm mailutils.scm.in

DOT_X_FILES=\
 mu_address.x\
 mu_body.x\
 mu_mailbox.x\
 mu_message.x\
 mu_mime.x\
 mu_logger.x\
 mu_port.x\
 mu_scm.x\
 mu_util.x

DOT_DOC_FILES=\
 mu_address.doc\
 mu_body.doc\
 mu_mailbox.doc\
 mu_message.doc\
 mu_mime.doc\
 mu_logger.doc\
 mu_port.doc\
 mu_scm.doc\
 mu_util.doc 

CLEANFILES=
DISTCLEANFILES=\
 mailutils.scm

mailutils.scm: mailutils.scm.in
	$(AM_V_GEN)m4 -DVERSION=$(VERSION) -DLIBDIR=$(libdir) \
	   -DSITEDIR=$(sitedir) \
           -DBUILDDIR=$(top_builddir)/libmu_scm \
            $(srcdir)/mailutils.scm.in > $@

install-data-hook:
	 @here=`pwd`; \
	 cd $(DESTDIR)$(libdir);\
	 if test -f libmu_scm.la; then \
		dlname=`sed -n 's/dlname='\''\(.*\)'\''/\1/p' libmu_scm.la`; \
		test -z "$dlname" && dlname='libmu_scm.so'; \
		$(LN_S) -f "$$dlname" libguile-mailutils-v-$(VERSION).so; \
	 fi; \
	 cd $$here

sitedir   = @GUILE_SITE@/$(PACKAGE)
site_DATA = mailutils.scm
SUFFIXES=
BUILT_SOURCES=
include ../gint/gint.mk
