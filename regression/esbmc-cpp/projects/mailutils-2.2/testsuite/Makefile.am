## Process this file with GNU Automake to create Makefile.in

##   Copyright (C) 2002, 2007, 2010 Free Software Foundation, Inc.
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

test_dirs = lib etc spool folder

dist-hook:
	@here=`cd $(top_builddir)/$(subdir) && pwd`; \
	srcdir=`cd $(srcdir) && pwd`; \
	distdir=`cd $(distdir) && pwd`; \
	for dir in $(test_dirs); \
	do \
            cd $$srcdir;\
            mkdir $$distdir/$$dir;\
	    cd $$dir;\
	    for file in DISTFILES `cat DISTFILES`; do \
	        d=$$srcdir/$$dir; \
	        if test -d $$d/$$file; then \
                    cp -pr $$d/$$file $$distdir/$$dir/$$file; \
		else \
	            test -f $$distdir/$$dir/$$file \
                        || cp -p $$d/$$file $$distdir/$$dir/$$file || exit; \
                fi; \
	    done;\
        done;\
	cd $$here

DISTCLEANFILES=*.exp *.log *.sum 

distclean-local:
	-rm -rf	data
