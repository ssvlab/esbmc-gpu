## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2004, 2005, 2006, 2007, 2010 Free Software
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

INCLUDES = @MU_LIB_COMMON_INCLUDES@ -I${top_srcdir}/libproto/include 

lib_LTLIBRARIES = libmu_nntp.la
libmu_nntp_la_LDFLAGS=-version-info @VI_CURRENT@:@VI_REVISION@:@VI_AGE@
libmu_nntp_la_LIBADD = ${MU_LIB_MAILUTILS}
libmu_nntp_la_SOURCES = \
 nntp_article.c \
 nntp_body.c \
 nntp_carrier.c \
 nntp_connect.c \
 nntp_create.c \
 nntp_date.c \
 nntp_debug.c \
 nntp_destroy.c \
 nntp_disconnect.c \
 nntp_group.c \
 nntp_head.c \
 nntp_help.c \
 nntp_ihave.c \
 nntp_iterator.c \
 nntp_last.c \
 nntp_list_active.c \
 nntp_list_distribpats.c \
 nntp_list_distributions.c \
 nntp_list_extensions.c \
 nntp_list_newsgroups.c \
 nntp_list_times.c \
 nntp_mode_reader.c \
 nntp_newgroups.c \
 nntp_newnews.c \
 nntp_next.c \
 nntp_post.c \
 nntp_quit.c \
 nntp_readline.c \
 nntp_response.c \
 nntp_sendline.c \
 nntp_stat.c \
 nntp_stream.c \
 nntp_timeout.c \
 nntp0.h \
 url.c \
 folder.c \
 mbox.c
