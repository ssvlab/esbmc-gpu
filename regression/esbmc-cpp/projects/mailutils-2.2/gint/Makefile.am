# This file is part of Gint
# Copyright (C) 2010 Sergey Poznyakoff
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

EXTRA_DIST = \
 clexer.l\
 gint.m4\
 gint.mk\
 guile.m4\
 snarf-doc-filter\
 extract-exports

if GINT_COND_DOC
if GINT_COND_SNARF_DOC_FILTER
noinst_SCRIPTS = snarf-doc-filter
else
noinst_PROGRAMS = clexer
clexer_SOURCES = clexer.l
INCLUDES = @GINT_INCLUDES@
LDADD = @GINT_LDADD@
endif
endif

