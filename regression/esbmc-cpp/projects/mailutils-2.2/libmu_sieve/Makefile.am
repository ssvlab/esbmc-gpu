## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2002, 2003, 2004, 2005, 2007, 2009, 2010 Free
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
##   Foundation, Inc.  51 Franklin Street, Fifth Floor, Boston, MA
##   02110-1301 USA

INCLUDES = @MU_LIB_COMMON_INCLUDES@ 

YLWRAP = $(SHELL) $(mu_aux_dir)/gylwrap
AM_YFLAGS = -dtv
AM_CPPFLAGS = -DMU_SIEVE_MODDIR=\"@MU_SIEVE_MODDIR@\"

SUBDIRS = . extensions

lib_LTLIBRARIES = libmu_sieve.la

libmu_sieve_la_SOURCES = \
 actions.c\
 conf.c\
 comparator.c\
 load.c\
 prog.c\
 register.c\
 relational.c\
 require.c\
 runtime.c\
 sieve-gram.c\
 sieve-gram.h\
 sieve-lex.c\
 tests.c\
 util.c
libmu_sieve_la_LIBADD = ${MU_LIB_MAILUTILS} @LTDL_LIB@
libmu_sieve_la_LDFLAGS = -version-info @VI_CURRENT@:@VI_REVISION@:@VI_AGE@

noinst_HEADERS = sieve-priv.h

BUILT_SOURCES = \
  sieve-gram.c sieve-gram.h \
  sieve-lex.c

MAINTAINERCLEANFILES=$(BUILT_SOURCES)

EXTRA_DIST = sieve.y sieve.l

sieve-lex.c: $(srcdir)/sieve.l sieve-gram.h
	$(YLWRAP) "$(LEX) $(AM_LEXFLAGS) $(LEXFLAGS)" \
                  $(srcdir)/sieve.l lex.yy.c sieve-lex.c \
                  -- -yy mu_sieve_yy

sieve-gram.c sieve-gram.h: $(srcdir)/sieve.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS) $(YFLAGS)" $(srcdir)/sieve.y \
	          y.tab.c sieve-gram.c y.tab.h sieve-gram.h y.output y.output \
                  -- -yy mu_sieve_yy


