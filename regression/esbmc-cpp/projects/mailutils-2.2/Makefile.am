## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
##   2009, 2010 Free Software Foundation, Inc.
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

ACLOCAL_AMFLAGS = -I m4 -I am -I gint

if MU_COND_PYTHON
  PYTHON_DIR = python
endif

if MU_COND_MH
  MH_DIR = mh
endif

if MU_COND_FRM
  FRM_DIR = frm
endif

if MU_COND_POP3D
  POP3D_DIR = pop3d
endif

if MU_COND_IMAP4D
  IMAP4D_DIR = imap4d
endif

if MU_COND_MAIDAG
  MAIDAG_DIR = maidag
endif

if MU_COND_MAIL
  MAIL_DIR = mail
endif

if MU_COND_SIEVE
  SIEVE_DIR = sieve
endif

if MU_COND_GUIMB
  GUIMB_DIR = guimb
endif

if MU_COND_MESSAGES
  MESSAGES_DIR = messages
endif

if MU_COND_COMSAT
  COMSAT_DIR = comsat
endif

if MU_COND_READMSG
  READMSG_DIR = readmsg
endif

if MU_COND_DOTLOCK
  DOTLOCK_DIR = dotlock
endif

if MU_COND_MOVEMAIL
  MOVEMAIL_DIR = movemail
endif

if MU_COND_MIMEVIEW
  MIMEVIEW_DIR = mimeview
endif

if MU_COND_LIBMU_SCM
  GINT_DIR = gint
  LIBMU_SCM_DIR = libmu_scm
endif

if MU_COND_LIBMU_CPP
  LIBMU_CPP_DIR = libmu_cpp
endif

SUBDIRS = \
 mu-aux\
 include\
 po\
 testsuite\
 mailbox\
 sql\
 libmu_auth\
 libproto\
 lib\
 libmu_argp\
 libmu_cfg\
 $(LIBMU_CPP_DIR)\
 $(GINT_DIR)\
 $(LIBMU_SCM_DIR)\
 libmu_sieve\
 $(PYTHON_DIR)\
 doc\
 config\
 examples\
 $(FRM_DIR)\
 $(POP3D_DIR)\
 $(IMAP4D_DIR)\
 $(MAIDAG_DIR)\
 $(MAIL_DIR)\
 $(SIEVE_DIR)\
 $(GUIMB_DIR)\
 $(MESSAGES_DIR)\
 $(COMSAT_DIR)\
 $(READMSG_DIR)\
 $(DOTLOCK_DIR)\
 $(MH_DIR)\
 $(MOVEMAIL_DIR)\
 $(MIMEVIEW_DIR)

EXTRA_DIST = COPYING.LESSER paths

DISTCLEANFILES = pathdefs.h

gen_start_date = "2008-12-08"
prev_change_log = "doc/ChangeLog.CVS"

.PHONY: ChangeLog
ChangeLog:
	@if test -d .git; then                                             \
	  cmd=$(top_srcdir)/build-aux/gitlog-to-changelog;                 \
	  if test -n "$(gen_start_date)"; then                             \
	    cmd="$$cmd --since=\"$(gen_start_date)\"";                     \
	  fi;                                                              \
	  $$cmd --format='%s%n%n%b%n' |                                    \
            sed '/<unknown>$$/d' | fmt -s > cl-t;                          \
          if test -n "$(prev_change_log)" && test -f "$(prev_change_log)"; \
	  then                                                             \
	    echo "" >> cl-t;                                               \
	    cat "$(prev_change_log)" |                                     \
	      sed '/^Local Variables:/,/^End:/d' >> cl-t;                  \
	  fi;                                                              \
	  echo "Local Variables:" >> cl-t;                                 \
	  echo "mode: change-log" >> cl-t;                                 \
	  echo "version-control: never"  >> cl-t;                          \
	  echo "buffer-read-only: t" >> cl-t;                              \
	  echo "End:" >> cl-t;                                             \
	  rm -f ChangeLog;                                                 \
	  mv cl-t ChangeLog;                                               \
	fi

alpha:
	$(MAKE) dist distdir=$(PACKAGE)-$(VERSION)-`date +"%Y%m%d"`

alphacheck:
	$(MAKE) distcheck distdir=$(PACKAGE)-$(VERSION)-`date +"%Y%m%d"`

rpm: dist
	rpm -ta --clean mailutils-$(VERSION).tar.gz

