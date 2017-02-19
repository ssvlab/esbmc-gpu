# This file is part of GNU Mailutils
# Copyright (C) 2007, 2009, 2010 Free Software Foundation, Inc.
#
# GNU Mailutils is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or (at
# your option) any later version.
#
# GNU Mailutils is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Mailutils; if not, write to the Free Software
# Foundation, Inc.  51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301 USA

INCLUDES = @MU_LIB_COMMON_INCLUDES@ 
AM_CFLAGS = -DSITE_VIRTUAL_PWDDIR=\"@SITE_VIRTUAL_PWDDIR@\" \
            -DSITE_CRAM_MD5_PWD=\"@SITE_CRAM_MD5_PWD@\"

lib_LTLIBRARIES = libmu_cfg.la
libmu_cfg_la_SOURCES=\
 acl.c\
 auth.c\
 common.c\
 init.c\
 gsasl.c\
 ldap.c\
 pam.c\
 radius.c\
 sieve.c\
 sql.c\
 tls.c\
 virtdomain.c
