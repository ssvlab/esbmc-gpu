## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2001, 2002, 2003, 2004, 2006, 2007, 2009, 2010 Free
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

info_TEXINFOS = mailutils.texi 

INCFILES = \
 addr.inc\
 http.inc\
 mailcap.inc\
 numaddr.inc\
 sfrom.inc\
 url-parse.inc

RENDITION_TEXI=rendition.texi macros.texi

mailutils_TEXINFOS = \
 address.texi\
 attribute.texi\
 auth.texi\
 body.texi\
 c-api.texi\
 encoding.texi\
 envelope.texi\
 fdl.texi\
 folder.texi\
 framework.texi\
 getdate.texi\
 headers.texi\
 imap4.texi\
 iterator.texi\
 libmu_scm.texi\
 libmu_auth.texi\
 libmu_sieve.texi\
 locker.texi\
 mailbox.texi\
 mailcap.texi\
 maildir.texi\
 mailer.texi\
 mbox.texi\
 message.texi\
 mh.texi\
 mu-mh.texi\
 mu_address.texi\
 mu_body.texi\
 mu_logger.texi\
 mu_mailbox.texi\
 mu_message.texi\
 mu_mime.texi\
 mu_scm.texi\
 nntp.texi\
 parse822.texi\
 pop3.texi\
 programs.texi\
 sendmail.texi\
 sieve.texi\
 smtp.texi\
 stream.texi\
 usage.texi\
 url.texi\
 $(RENDITION_TEXI)\
 $(INCFILES)

## Fake configure into including srcdir to VPATH:
s=${srcdir}:${top_srcdir}/examples/
VPATH = $(s)

SUFFIXES=.c .inc 
.c.inc:
	sed -f $(mu_aux_dir)/texify.sed $< > $@

DISTCLEANFILES=*.pgs *.kys *.vrs
MAINTAINERCLEANFILES=$(INCFILES)
clean-local:
	rm -rf manual

# The rendering level is one of PUBLISH, DISTRIB or PROOF.
# Just call `make RENDITION=PROOF [target]' if you want PROOF rendition.

MAKEINFOFLAGS=-D$(RENDITION)
GENDOCS=gendocs.sh
TEXI2DVI=texi2dvi -t '@set $(RENDITION)' -E

# Make sure you set TEXINPUTS.
# TEXINPUTS=/usr/share/texmf/pdftex/plain/misc/ is ok for most distributions

TEXINPUTS=$(srcdir):$(top_srcdir)/build-aux:$$TEXINPUTS

manual:
	 TEXINPUTS=$(srcdir):$(top_srcdir)/build-aux:$(TEXINPUTS) \
	 MAKEINFO="$(MAKEINFO) $(MAKEINFOFLAGS)" \
	 TEXI2DVI="$(TEXI2DVI) -t @finalout" \
	$(GENDOCS) $(PACKAGE) '$(PACKAGE_NAME) manual'


EXTRA_DIST=gendocs_template mastermenu.el untabify.el

master-menu:
	emacs -batch -l mastermenu.el -f make-master-menu $(info_TEXINFOS)

untabify:
	emacs -batch -l untabify.el $(info_TEXINFOS) $(mailutils_TEXINFOS)

fix-sentence-spacing:
	for file in $(info_TEXINFOS) $(mailutils_TEXINFOS); \
	do \
		if grep -q '\.  [@A-Z]' $$file; then \
			mv $$file $${file}~; \
			sed -r 's/\.  ([@A-Z])/. \1/g' $${file}~ > $$file; \
		fi; \
	done

final: untabify fix-sentence-spacing master-menu


# Checks
check-tabs:
	@if test -n "`cat $(info_TEXINFOS) $(mailutils_TEXINFOS) |\
             tr -d -c '\t'`"; then \
		echo >&2 "Sources contain tabs; run make untabify"; \
		false; \
	fi

check-sentence-spacing:
	@if grep -q '\.  [@A-Z]' $(info_TEXINFOS) $(mailutils_TEXINFOS); then \
		echo >&2 "Sources contain double-space sentence separators"; \
		echo >&2 "Run make fix-sentence-spacing to fix"; \
	fi

check-format: check-tabs check-sentence-spacing

check-refs:
	@for file in $(info_TEXINFOS) $(mailutils_TEXINFOS); \
	do \
	  sed -e = $$file | \
           sed -n 'N;/@FIXME-.*ref/{s/\(^[0-9][0-9]*\).*@FIXME-.*ref{\([^}]*\)}.*/'$$file':\1: \2/gp}'; \
	done > $@-t; \
	if [ -s $@-t ]; then \
	  echo >&2 "Unresolved cross-references:"; \
	  cat $@-t >&2;\
	  rm $@-t; \
	else \
	  rm -f $@-t; \
	fi

check-fixmes:
	@for file in $(info_TEXINFOS) $(mailutils_TEXINFOS); \
	do \
	  sed -e = $$file | \
           sed -n 'N;/@FIXME{/{s/\(^[0-9][0-9]*\).*@FIXME{\([^}]*\).*/'$$file':\1: \2/gp}'; \
	done > $@-t; \
	if [ -s $@-t ]; then \
	  echo >&2 "Unresolved FIXMEs:"; \
	  cat $@-t >&2; \
	  rm $@-t; \
	  false; \
	else \
          rm -f $@-t; \
	fi

check-writeme:
	@grep -Hn @WRITEME $(info_TEXINFOS) $(mailutils_TEXINFOS) > $@-t; \
	if [ -s $@-t ]; then \
	  echo "Empty nodes:"; \
	  cat $@-t; \
	  rm $@-t; \
	  false;\
	else \
          rm $@-t; \
	fi

check-unrevised:
	@grep -Hn @UNREVISED $(info_TEXINFOS) $(mailutils_TEXINFOS) > $@-t; \
	if [ -s $@-t ]; then \
	  echo >&2 "Unrevised nodes:"; \
	  cat $@-t >&2; \
	  rm $@-t; \
	  false;\
	else \
          rm $@-t; \
	fi

all-check-docs: check-format check-writeme check-unrevised check-refs check-fixmes 

check-docs: 
	$(MAKE) -k all-check-docs
