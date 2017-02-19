# Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.
#
# GNU Mailutils is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc.  51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301 USA

INCLUDES = -I${top_srcdir} @MU_APP_COMMON_INCLUDES@ @GUILE_INCLUDES@ \
 @PYTHON_INCLUDES@

sbin_PROGRAMS=maidag
maidag_SOURCES=\
 deliver.c\
 forward.c\
 guile.c\
 lmtp.c\
 maidag.c\
 maidag.h\
 mailtmp.c\
 mailquota.c\
 python.c\
 sieve.c\
 script.c\
 util.c

maidag_LDADD = \
 @LIBMU_SCM@ @GUILE_LIBS@\
 @LIBMU_SCM_DEPS@\
 @MU_LIB_PY@ @PYTHON_LIBS@\
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_SIEVE}\
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 ${MU_LIB_MAILER}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS} \
 @MU_COMMON_LIBRARIES@\
 @TCPWRAP_LIBRARIES@

install-exec-hook:
	for i in $(sbin_PROGRAMS); do\
		chown root:mail $(DESTDIR)$(sbindir)/$$i;\
		chmod 4755 $(DESTDIR)$(sbindir)/$$i;\
	done
