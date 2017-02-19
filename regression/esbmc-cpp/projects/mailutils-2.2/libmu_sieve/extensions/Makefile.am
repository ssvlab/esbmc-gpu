## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2004, 2005, 2006, 2007, 2009, 2010 Free Software
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

moddir=@MU_SIEVE_MODDIR@
mod_LTLIBRARIES = \
 list.la\
 moderator.la\
 pipe.la\
 spamd.la\
 timestamp.la\
 vacation.la

INCLUDES = @MU_APP_COMMON_INCLUDES@ 

AM_LDFLAGS = -module -avoid-version -no-undefined -rpath '$(moddir)' 

list_la_SOURCES = list.c
LIBS = ../libmu_sieve.la 

moderator_la_SOURCES = moderator.c

spamd_la_SOURCES = spamd.c

timestamp_la_SOURCES = timestamp.c

vacation_la_SOURCES = vacation.c

pipe_la_SOURCES = pipe.c
