## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2001, 2002, 2003, 2007, 2009, 2010 Free Software
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

bindir = @MH_BIN_DIR@
mhlibdir = $(pkgdatadir)/mh
bin_PROGRAMS = \
 ali\
 anno\
 burst\
 comp\
 fmtcheck\
 folder\
 forw\
 inc\
 install-mh\
 mark\
 mhl\
 mhn\
 mhparam\
 mhpath\
 pick\
 refile\
 repl\
 rmf\
 rmm\
 scan\
 send\
 sortm\
 whatnow\
 whom

noinst_LIBRARIES = libmh.a

libmh_a_SOURCES= \
 compcommon.c\
 mh_alias_gram.c\
 mh_alias_lex.c\
 mh_argp.c\
 mh_ctx.c\
 mh_getopt.c\
 mh_global.c\
 mh_format.c\
 mh_init.c\
 mh_list.c\
 mh_fmtgram.c\
 mh_msgset.c\
 mh_sequence.c\
 mh_stream.c\
 mh_whatnow.c\
 mh_whom.c
noinst_HEADERS = mh.h mh_alias.h mh_format.h mh_getopt.h

LISPSRC = mailutils-mh.el
lisp_LISP=@lisp_LISP@
EXTRA_LISP = mailutils-mh.el

BUILT_SOURCES= \
	mh_fmtgram.c \
        pick-gram.c \
        pick-gram.h \
        mh_alias_gram.c \
        mh_alias_lex.c \
        mh_alias.h

MAINTAINERCLEANFILES=$(BUILT_SOURCES)

mhlib_DATA = components replcomps replgroupcomps mhl.format $(LISPSRC)
EXTRA_DIST = components replcomps replgroupcomps mhl.format\
 mailutils-mh.eli mh_fmtgram.y pick.y mh_alias.y mh_alias.l
DISTCLEANFILES = mailutils-mh.el

SUFFIXES = .eli .el
mailutils-mh.el: mailutils-mh.eli

.eli.el:
	sed "s,BINDIR,$(bindir),g;s,MHLIBDIR,$(mhlibdir),g" $< > $@

INCLUDES = @MU_APP_COMMON_INCLUDES@ 
AM_CPPFLAGS = -D_GNU_SOURCE -DMHLIBDIR=\"$(mhlibdir)\" -DMHBINDIR=\"$(bindir)\"
mh_LIBS = \
 ./libmh.a\
 ../lib/libmuaux.la\
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_MH}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_MAILER}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}\
 @MU_COMMON_LIBRARIES@
LDADD = $(mh_LIBS)
scan_LDADD = $(mh_LIBS) @CURSES_LIBS@
pick_SOURCES = pick.c pick.h pick-gram.c pick-gram.h

YLWRAP = $(SHELL) $(mu_aux_dir)/gylwrap
AM_YFLAGS=-vt

pick-gram.c pick-gram.h: $(srcdir)/pick.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS) -d" $< \
	          y.tab.c pick-gram.c y.tab.h pick-gram.h \
                  y.output pick.output \
                  -- -yy pick_yy

mh_fmtgram.c: $(srcdir)/mh_fmtgram.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS)" $< \
	          y.tab.c mh_fmtgram.c y.output fmtgram.y.output \
                  -- -yy fmt_yy

mh_alias_gram.c mh_alias.h: $(srcdir)/mh_alias.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS) -d" $< \
	          y.tab.c mh_alias_gram.c y.tab.h mh_alias.h \
                  y.output mh_alias.output \
                  -- -yy ali_yy

mh_alias_lex.c: $(srcdir)/mh_alias.l mh_alias.h
	$(YLWRAP) "$(LEX) $(AM_LEXFLAGS) $(LEXFLAGS)" \
                  $(srcdir)/mh_alias.l lex.yy.c mh_alias_lex.c \
                  -- -yy ali_yy

install-exec-hook: 
	@here=`pwd`; \
	cd $(DESTDIR)$(bindir); \
	rm -f folders; \
	$(LN_S) folder folders; \
	cd $$here
