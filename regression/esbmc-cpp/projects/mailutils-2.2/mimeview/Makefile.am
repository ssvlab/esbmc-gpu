## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2005, 2007, 2010 Free Software Foundation, Inc.
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

INCLUDES = @MU_APP_COMMON_INCLUDES@ 

AM_CFLAGS = -DDEFAULT_CUPS_CONFDIR=\"@DEFAULT_CUPS_CONFDIR@\"

bin_PROGRAMS = mimeview
mimeview_SOURCES = \
 mimeview.c \
 mimetypes-gram.c \
 mimetypes-lex.c \
 mimetypes-decl.h \
 mimeview.h 

YLWRAP = $(SHELL) $(mu_aux_dir)/gylwrap
AM_YFLAGS=-vt
AM_LEXFLAGS=-d
AM_CPPFLAGS=-D_GNU_SOURCE=1
EXTRA_DIST = mimetypes.y mimetypes.l

mimetypes-gram.c mimetypes-decl.h: $(srcdir)/mimetypes.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS) -d" $< \
	          y.tab.c mimetypes-gram.c y.tab.h mimetypes-decl.h \
                  y.output mimetypes.output \
                  -- -yy mimetypes_yy

mimetypes-lex.c: $(srcdir)/mimetypes.l mimetypes-decl.h
	$(YLWRAP) "$(LEX) $(AM_LEXFLAGS) $(LEXFLAGS)" \
                  $(srcdir)/mimetypes.l lex.yy.c mimetypes-lex.c \
                  -- -yy mimetypes_yy

BUILT_SOURCES = mimetypes-gram.c mimetypes-lex.c mimetypes-decl.h

mimeview_LDADD = \
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_MAILUTILS}\
 @LTLIBINTL@
 
