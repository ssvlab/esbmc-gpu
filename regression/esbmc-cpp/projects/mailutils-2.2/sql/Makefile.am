## This file is part of GNU Mailutils

##   Copyright (C) 2004, 2005, 2007, 2010 Free Software Foundation, Inc.
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

noinst_LTLIBRARIES=@BUILD_SQL@
EXTRA_LTLIBRARIES=libsql.la mysql.la postgres.la odbc.la
EXTRA_DIST=\
 mysql.c\
 odbc.c\
 postgres.c
DISTCLEANFILES = modlist.h

INCLUDES= @MU_LIB_COMMON_INCLUDES@ 
libsql_la_SOURCES=sql.c
libsql_la_LIBADD=@SQL_LTLIBOBJS@

libsql_la_DEPENDENCIES = @SQL_LTLIBOBJS@

moddir = $(pkgdatadir)/$(VERSION)/modules
mod_LTLIBRARIES = @SQL_LOADABLE_MODULES@

MODFLAGS=-module -avoid-version -no-undefined -rpath $(moddir) -Wl,-Bsymbolic

mysql_la_SOURCES = mysql.c
mysql_la_LIBADD = @MYSQLLIBS@
mysql_la_LDFLAGS = $(MODFLAGS)

postgres_la_SOURCES = postgres.c
postgres_la_LIBADD = @POSTGRESLIBS@
postgres_la_LDFLAGS = $(MODFLAGS)

odbc_la_SOURCES = odbc.c
odbc_la_LIBADD = @ODBCLIBS@
odbc_la_LDFLAGS = $(MODFLAGS)
