## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2001, 2002, 2006, 2007, 2009, 2010 Free Software
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

MU_GUILE_SIEVE_SCRIPTS_X=sieve.scm
bin_SCRIPTS = sieve.scm
EXTRA_SCRIPTS=sieve.scm

sievemoddir=@MU_GUILE_SIEVE_MOD_DIR@

sieve.scm: sieve.scm.in sieve.sed
	$(AM_V_GEN)sed -f sieve.sed $(srcdir)/sieve.scm.in > sieve.scm

sieve.sed: Makefile
	$(AM_V_GEN)echo 's,%GUILE_BINDIR%,@GUILE_BINDIR@,g' > sieve.sed
	$(AM_V_at)echo 's,%BINDIR%,$(bindir),g' >> sieve.sed
	$(AM_V_at)echo 's,%GUILE_SITE%,$(GUILE_SITE),g' >> sieve.sed
	$(AM_V_at)echo 's,%LIBDIR%,$(sievemoddir),g' >> sieve.sed
	$(AM_V_at)echo 's,%PACKAGE%,$(PACKAGE),g' >> sieve.sed
	$(AM_V_at)echo 's,%VERSION%,$(VERSION),g' >> sieve.sed

CLEANFILES = sieve.scm sieve.sed

sitedir=@GUILE_SITE@/$(PACKAGE)
site_DATA=sieve-core.scm

MU_GUILE_SIEVE_MOD_DATA_X=\
 mimeheader.scm\
 numaddr.scm\
 redirect.scm\
 reject.scm\
 vacation.scm

sievemod_DATA=@MU_GUILE_SIEVE_MOD_DATA@

EXTRA_DIST=\
 mimeheader.scm\
 numaddr.scm\
 redirect.scm\
 reject.scm\
 sieve-core.scm\
 sieve.scm.in\
 vacation.scm

