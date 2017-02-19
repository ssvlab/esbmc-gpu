## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2009, 2010 Free Software Foundation, Inc.
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

INCLUDES = @MU_LIB_COMMON_INCLUDES@ $(PYTHON_INCLUDES)

lib_LTLIBRARIES=libmu_py.la
libmu_py_la_SOURCES = \
 error.c \
 address.c \
 attribute.c \
 auth.c \
 body.c \
 debug.c \
 envelope.c \
 filter.c \
 folder.c \
 libmu_py.c \
 libmu_py.h \
 list.c \
 header.c \
 mailer.c \
 mailbox.c \
 mailcap.c \
 message.c \
 mime.c \
 nls.c \
 script.c \
 secret.c \
 sieve.c \
 stream.c \
 registrar.c \
 url.c \
 util.c
libmu_py_la_LDFLAGS = -rpath $(libdir) \
 -export-symbols-regex '^(mu_py_|Py).*' \
 -version-info @VI_CURRENT@:@VI_REVISION@:@VI_AGE@
libmu_py_la_LIBADD = $(PYTHON_LIBS) @MU_COMMON_LIBRARIES@ \
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_AUTH}\
 ${MU_LIB_MAILER}\
 ${MU_LIB_SIEVE}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}

pythonexecdir=@PYTHON_EXEC_DIR@
pythonexec_LTLIBRARIES = c_api.la
c_api_la_LDFLAGS = -avoid-version -module -rpath $(pythonexecdir)
c_api_la_LIBADD = $(PYTHON_LIBS) ${MU_LIB_PY}
c_api_la_SOURCES = c_api.c
