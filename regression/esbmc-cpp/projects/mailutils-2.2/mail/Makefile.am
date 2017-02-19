## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 1999, 2000, 2001, 2002, 2005, 2007, 2010 Free
##   Software Foundation, Inc.
##
##   GNU Mailutils is free software; you can redistribute it and/or
##   modify it under the terms of the GNU General Public License as
##   published by the Free Software Foundation; either version 2, or (at
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

AM_YFLAGS=-vt
INCLUDES = @MU_APP_COMMON_INCLUDES@ @MU_AUTHINCS@ 
AM_CFLAGS = -DSITE_MAIL_RC=\"@SITE_MAIL_RC@\"

bin_PROGRAMS = mail

mail_LDADD = \
 ${MU_APP_LIBRARIES}\
 ${MU_LIB_MBOX}\
 ${MU_LIB_IMAP}\
 ${MU_LIB_POP}\
 ${MU_LIB_NNTP}\
 ${MU_LIB_MH}\
 ${MU_LIB_MAILDIR}\
 ${MU_LIB_MAILER}\
 ${MU_LIB_AUTH}\
 @MU_AUTHLIBS@\
 ${MU_LIB_MAILUTILS}\
 @READLINE_LIBS@ @MU_COMMON_LIBRARIES@

EXTRA_DIST = msgset.c

mail_SOURCES = \
 alias.c\
 alt.c\
 cd.c\
 copy.c\
 decode.c\
 delete.c\
 dp.c\
 echo.c\
 edit.c\
 envelope.c\
 eq.c\
 escape.c\
 exit.c\
 file.c\
 folders.c\
 followup.c\
 from.c\
 headers.c\
 help.c\
 hold.c\
 if.c\
 inc.c\
 list.c\
 mail.c\
 mail.h\
 mailline.c\
 mailvar.c\
 mbox.c\
 msgset.y\
 next.c\
 page.c\
 pipe.c\
 previous.c\
 print.c\
 quit.c\
 reply.c\
 retain.c\
 save.c\
 send.c\
 set.c\
 setenv.c\
 shell.c\
 size.c\
 source.c\
 struct.c\
 summary.c\
 table.c\
 tag.c\
 top.c\
 touch.c\
 unalias.c\
 undelete.c\
 unset.c\
 util.c\
 version.c\
 visual.c\
 write.c\
 z.c

SUBDIRS = testsuite

