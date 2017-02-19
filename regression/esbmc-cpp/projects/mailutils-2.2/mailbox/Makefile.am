## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009,
##   2010 Free Software Foundation, Inc.
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

INCLUDES = @MU_LIB_COMMON_INCLUDES@

YLWRAP = $(SHELL) $(mu_aux_dir)/gylwrap
AM_YFLAGS=-vt
AM_LFLAGS=-dvp

SUBDIRS = testsuite 

lib_LTLIBRARIES = libmailutils.la

localedir = $(datadir)/locale
AM_CPPFLAGS = \
 -DSYSCONFDIR=\"$(sysconfdir)\"\
 -DSITE_VIRTUAL_PWDDIR=\"@SITE_VIRTUAL_PWDDIR@\"\
 -DLOCALEDIR=\"$(localedir)\"

EXTRA_DIST = \
 errors\
 muerrno.cin\
 parsedate.y\
 fgetpwent.c\
 cfg_lexer.l\
 cfg_parser.y\
 cfg_parser.h

libmailutils_la_SOURCES = \
 acl.c\
 address.c\
 alloc.c\
 amd.c\
 argcv.c\
 asnprintf.c\
 asprintf.c\
 assoc.c\
 attachment.c\
 attribute.c\
 auth.c\
 base64.c\
 body.c\
 cstrcasecmp.c\
 cfg_driver.c\
 cfg_format.c\
 cfg_lexer.c\
 cfg_parser.c\
 cstrlower.c\
 cstrupper.c\
 daemon.c\
 date.c\
 dbgstderr.c\
 dbgsyslog.c\
 debug.c\
 diag.c\
 envelope.c\
 fgetpwent.c\
 file_stream.c\
 filter.c\
 filter_iconv.c\
 filter_rfc822.c\
 filter_trans.c\
 folder.c\
 freeitem.c\
 gdebug.c\
 gocs.c\
 hdritr.c\
 header.c\
 iterator.c\
 ipsrv.c\
 kwd.c\
 list.c\
 listlist.c\
 locale.c\
 locker.c\
 mailbox.c\
 mailcap.c\
 mailer.c\
 mapfile_stream.c\
 mbx_default.c\
 mbxitr.c\
 md5.c\
 message.c\
 memory_stream.c\
 message_stream.c\
 mime.c\
 mimehdr.c\
 mkfilename.c\
 monitor.c\
 msrv.c\
 mu_auth.c\
 muctype.c\
 munre.c\
 mutil.c\
 muerror.c\
 muerrno.c\
 nls.c\
 observer.c\
 opool.c\
 parse822.c\
 parsedate.c\
 permstr.c\
 progmailer.c\
 property.c\
 registrar.c\
 refcount.c\
 rfc2047.c\
 sha1.c\
 secret.c\
 server.c\
 socket_stream.c\
 stream.c\
 strltrim.c\
 strskip.c\
 stripws.c\
 strrtrim.c\
 syslog.c\
 system.c\
 ticket.c\
 tcp.c\
 url.c\
 vartab.c\
 vasnprintf.c\
 version.c\
 wicket.c
 
BUILT_SOURCES = parsedate.c muerrno.c cfg_parser.c cfg_parser.h cfg_lexer.c
MOSTLYCLEANFILES=

parsedate.c: $(srcdir)/parsedate.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS)" $< \
	          y.tab.c parsedate.c y.output parsedate.y.output \
                  -- -yy pd_yy

cfg_parser.c cfg_parser.h: $(srcdir)/cfg_parser.y
	$(YLWRAP) "$(YACC) $(AM_YFLAGS) -d" $< \
	          y.tab.c cfg_parser.c y.tab.h cfg_parser.h \
                  y.output cfg_parser.y.output \
                  -- -yy mu_cfg_yy

cfg_lexer.c: $(srcdir)/cfg_lexer.l cfg_parser.h
	$(YLWRAP) "$(LEX) $(AM_LFLAGS) $(LFLAGS)" \
                  $(srcdir)/cfg_lexer.l lex.yy.c cfg_lexer.c \
                  -- -yy mu_cfg_yy

muerrno.c: errors muerrno.cin
	$(AWK) -f $(mu_aux_dir)/generr.awk $^ > $@

libmailutils_la_LIBADD = @MU_COMMON_LIBRARIES@
libmailutils_la_LDFLAGS = -version-info @VI_CURRENT@:@VI_REVISION@:@VI_AGE@

