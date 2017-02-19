dnl Configuration for GNU Mailutils -- a suite of utilities for electronic mail
dnl
dnl Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
dnl 2008, 2009, 2010 Free Software Foundation, Inc.
dnl
dnl GNU Mailutils is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3, or (at your option)
dnl any later version.
dnl
dnl GNU Mailutils is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License along
dnl with GNU Mailutils; if not, write to the Free Software Foundation,
dnl Inc. 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

AC_PREREQ(2.63)
AC_INIT([GNU Mailutils], [2.2], [bug-mailutils@gnu.org], [mailutils])
AC_CONFIG_SRCDIR([mailbox/mailbox.c])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([gnits 1.11 dist-bzip2 dist-lzma dist-xz std-options silent-rules])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Enable silent rules by default:
AM_SILENT_RULES([yes])

dnl Autobuild stuff
AB_INIT

dnl Library versioning
AC_SUBST(VI_CURRENT,  3)
AC_SUBST(VI_REVISION, 2)
AC_SUBST(VI_AGE,      1)

dnl Library paths
AC_SUBST(MU_LIB_AUTH,'${top_builddir}/libmu_auth/libmu_auth.la')
AC_SUBST(MU_LIB_MAILUTILS,'${top_builddir}/mailbox/libmailutils.la')
AC_SUBST(MU_LIB_SQL,'${top_builddir}/sql/libsql.la')
AC_SUBST(MU_LIB_IMAP,'${top_builddir}/libproto/imap/libmu_imap.la')
AC_SUBST(MU_LIB_MAILDIR,'${top_builddir}/libproto/maildir/libmu_maildir.la')
AC_SUBST(MU_LIB_MAILER,'${top_builddir}/libproto/mailer/libmu_mailer.la')
AC_SUBST(MU_LIB_MBOX,'${top_builddir}/libproto/mbox/libmu_mbox.la')
AC_SUBST(MU_LIB_MH,'${top_builddir}/libproto/mh/libmu_mh.la')
AC_SUBST(MU_LIB_NNTP,'${top_builddir}/libproto/nntp/libmu_nntp.la')
AC_SUBST(MU_LIB_POP,'${top_builddir}/libproto/pop/libmu_pop.la')
AC_SUBST(MU_LIB_SIEVE,'${top_builddir}/libmu_sieve/libmu_sieve.la')
AC_SUBST(MU_LIB_SCM,'${top_builddir}/libmu_scm/libmu_scm.la')
AC_SUBST(MU_LIB_CPP,'${top_builddir}/libmu_cpp/libmu_cpp.la')
AC_SUBST(MU_LIB_ARGP,'${top_builddir}/libmu_argp/libmu_argp.la')
AC_SUBST(MU_LIB_PY)

dnl Other variables
AC_SUBST(mu_aux_dir,'$(top_srcdir)/mu-aux')
AC_SUBST(MU_SIEVE_MODDIR,'$(libdir)/$(PACKAGE)')

AC_SUBST(MU_COMMON_LIBRARIES,'$(LTLIBINTL) $(LTLIBICONV)')
AC_SUBST(MU_APP_LIBRARIES,'../libmu_argp/libmu_argp.a ../libmu_cfg/libmu_cfg.la ../lib/libmuaux.la')

# There are two sets of include directories: MU_LIB_COMMON_INCLUDES, used
# by the libraries, and MU_APP_COMMON_INCLUDES, which is used by applications.
# The principal difference between them is that the latter contains gnulib
# directory.
#
# Libraries are isolated from gnulib and are not allowed to include its
# header files. There are two exceptions, though. See NOTE before the
# AC_CONFIG_LINKS statements, below.

AC_SUBST(MU_LIB_COMMON_INCLUDES,'-I${top_builddir} -I${top_srcdir}/include  -I${top_builddir}/include -I${top_srcdir}/mailbox -I${top_srcdir}/libproto/include')

AC_SUBST(MU_APP_COMMON_INCLUDES,'-I${srcdir} -I${top_srcdir}/include -I${top_srcdir}/lib -I${top_builddir}/lib -I${top_builddir} -I${top_builddir}/include -I${top_srcdir}/mailbox -I${top_srcdir}/libmu_argp -I${top_srcdir}/libmu_cfg')

dnl Check for programs
AC_PROG_CC
AM_PROG_CC_C_O
gl_EARLY
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL
AC_PROG_YACC
AM_PROG_LEX
AM_PROG_LIBTOOL

## Predefine several variables used to display configuration status
status_pam=no
status_ltdl=no
status_dbm=no
status_gsasl=no
status_mysql=no
status_pgsql=no
status_ldap=no
status_tcpwrap=maybe

dnl Internationalization macros.
AM_GNU_GETTEXT([external], [need-ngettext])
AM_GNU_GETTEXT_VERSION(0.17)
AM_ICONV
AC_CHECK_HEADERS(iconv.h)
AC_CHECK_TYPE(iconv_t,:,
              AC_DEFINE(iconv_t, int,
                        [Conversion descriptor type]),
              [
#ifdef HAVE_ICONV_H
# include <iconv.h>
#endif
])

# NOTE: Generally speaking, Mailutils libraries are not allowed to include
# gnulib headers.  The only exceptions are: gettext.h, which is needed for
# proper i18n, and intprops.h, which is used in mailbox/cfg_parser.y.
# This latter will probably be rewritten to remove the need for intprops.h.
#
# To make these two headers accessible for the libraries, they are linked
# to include/.
#
AC_CONFIG_LINKS(include/gettext.h:lib/gettext.h)
AC_CONFIG_LINKS(include/intprops.h:lib/intprops.h)

AH_BOTTOM([
#define DEFAULT_TEXT_DOMAIN PACKAGE
])

AC_SUBST(MU_LINK_POSTFLAGS,[`for opt in $LTLIBICONV; do case $opt in -[[lL]]*) echo $opt;; esac; done|tr '\n' ' '`])


dnl Escape defuns that force automake to add their arguments
dnl to LIBOBJS.
dnl All dependencies has been handled by gnulib-sync.

AC_DEFINE([GNULIB_ARGP_EXTERN_BASENAME],1,
          [Define to use our own version of __argp_base_name])

# Check for large file support
AC_SYS_LARGEFILE

gl_INIT

dnl Check for libraries
USE_LOADABLE_MODULES=no
AC_CHECK_LIB(resolv, res_query)
LTDL_LIB=
AC_SUBST(LTDL_LIB)
if test "$enable_shared" = yes; then
  AC_CHECK_LIB(ltdl, lt_dlinit,[
    AC_DEFINE(HAVE_LIBLTDL,
	      1,
	      [Define to 1 if you have the `ltdl' library (-lltdl).])
    status_ltdl=yes
    LTDL_LIB=-lltdl
    _USE_LOADABLE_MODULES=yes], # FIXME: Remove leading _ when ltd support
                                # is added to SQL.
    [LTDL_LIB=""])
fi
if test $USE_LOADABLE_MODULES = yes; then
  AC_DEFINE(USE_LOADABLE_MODULES,1,
            [Define if mailutils is able to use loadable modules])
fi

MU_DEBUG_MODE

# Initialize authlibs

AC_SUBST(MU_AUTHLIBS)
AC_SUBST(MU_AUTHLIBS_DEPENDENCY)
AC_SUBST(MU_AUTHINCS)

MU_AUTHLIBS=
MU_AUTHLIBS_DEPENDENCY=../libmu_auth/libmu_auth.la

dnl Check for arguments
AC_ARG_ENABLE([pam],
              AC_HELP_STRING([--disable-pam],
                             [disable pam]),
              [
case "${enableval}" in
  yes) testpam=yes ;;
  no)  testpam=no ;;
  *)   AC_MSG_ERROR([bad value ${enableval} for --disable-pam]) ;;
esac],[testpam=yes])

AC_ARG_WITH(tcp-wrappers,
	AC_HELP_STRING([--with-tcp-wrappers],
	               [compile with TCP wrappers (libwrap) support]),
	[status_tcpwrap=${withval}],
	[status_tcpwrap=maybe])

saved_LIBS=$LIBS
TCPWRAP_LIBRARIES=
case $status_tcpwrap in
 yes) AC_CHECK_LIB(wrap, main,,
                   [AC_MSG_ERROR([Required library libwrap not found])])
      AC_CHECK_LIB(nsl, main,
                   [TCPWRAP_LIBRARIES=-lnsl])
      AC_CHECK_HEADERS(tcpd.h,,
                       [AC_MSG_ERROR([Required header tcpd.h not found])])
      status_tcpwrap=yes
      ;;

 maybe)
      AC_CHECK_LIB(wrap, main,
                   [status_tcpwrap=yes],
                   [status_tcpwrap=no])
      AC_CHECK_LIB(nsl, main, [TCPWRAP_LIBRARIES=-lnsl])
      AC_CHECK_HEADERS(tcpd.h,
                       [status_tcpwrap=yes],
                       [status_tcpwrap=no])
      ;;
 no)  ;;
esac
LIBS=$saved_LIBS

if test "$status_tcpwrap" = "yes"; then
  AC_SUBST(TCPWRAP_LIBRARIES, "$TCPWRAP_LIBRARIES -lwrap")
  AC_DEFINE_UNQUOTED(WITH_LIBWRAP, 1,
                     [Define to 1 to use tcp wrappers.])
fi

AC_ARG_ENABLE([pthread],
              AC_HELP_STRING([--disable-pthread],
                             [disable pthread]),
              [
case "${enableval}" in
  yes) usepthread=yes ;;
  no)  usepthread=no ;;
  *)   AC_MSG_ERROR(bad value ${enableval} for --disable-pthread) ;;
esac],[usepthread=yes])

AC_ARG_WITH([readline],
            AC_HELP_STRING([--without-readline],
                           [do not use readline]),
            [
case "${withval}" in
  yes) usereadline=yes ;;
  no)  usereadline=no ;;
  *)   AC_MSG_ERROR(bad value ${withval} for --without-readline) ;;
esac],[usereadline=yes])

AC_SUBST(IMAP_AUTHOBJS)
AC_SUBST(SERV_AUTHLIBS)
AC_SUBST(SERV_AUTHINCS)

AC_ARG_WITH([gssapi],
            AC_HELP_STRING([--with-gssapi],
                           [use GSSAPI authentication]),
            [WITH_GSSAPI=$withval],
            [WITH_GSSAPI=no])

if test "x$WITH_GSSAPI" != "xno"; then
  MU_CHECK_GSSAPI($WITH_GSSAPI)
  if test "$GSSAPI_IMPL" = "none"; then
    AC_CACHE_SAVE
    AC_MSG_RESULT([GSSAPI libraries not found])
  else
    # Output produced by `krb5-config --libs' may contain `-Wl,-rpath -Wl,DIR'
    # options. Once they are added to SERV_AUTHLIBS, libtool, when creating
    # relink command line in wrapper files for executables, places them at the
    # beginnig of the option list and adds its own -rpath directives at
    # the end. Executables created by such command will look for the runtime
    # libraries first in DIR and then in the local tree subdirectories. Thus,
    # if DIR contains libraries from a previous installation of Mailutils, the
    # programs will use old libraries.
    #
    # Obviously, libtool should be fixed to place user-added -rpath directives
    # after the ones added by it. For the time being I work around this bug by
    # removing any -Wl directives from GSSAPI_LIBS.
    #
    for arg in $GSSAPI_LIBS
    do
      case $arg in
      -Wl,*) ;;
      *)     SERV_AUTHLIBS="$SERV_AUTHLIBS $arg"
      esac
    done
    SERV_AUTHINCS="$SERV_AUTHINCS $GSSAPI_CFLAGS"
    IMAP_AUTHOBJS="$IMAP_AUTHOBJS auth_gss.o"
    AC_DEFINE(WITH_GSSAPI,1,[Define if mailutils is using GSSAPI])
  fi
fi

MU_CHECK_GSASL(0.2.3, [
    AC_DEFINE(WITH_GSASL,1,[Define if mailutils is using GNU SASL])
    MU_AUTHLIBS="$MU_AUTHLIBS $GSASL_LIBS"
    IMAP_AUTHOBJS="$IMAP_AUTHOBJS auth_gsasl.o"
    status_gsasl=yes])

AC_SUBST(SITE_CRAM_MD5_PWD,"\$(sysconfdir)/cram-md5.pwd")

AC_ARG_WITH(cram-passwd,
            AC_HELP_STRING([--with-cram-passwd=FILE],
                           [use keep passwords for CRAM-MD5 authentication method in FILE (default $SITE_CRAM_MD5_PWD)]),
            [case "${withval}" in
            /*) SITE_CRAM_MD5_PWD="${withval}";;
            ./*|../*) AC_MSG_ERROR(Bad value for --with-cram-passwd);;
            *)  SITE_CRAM_MD5_PWD="\$(sysconfdir)/${withval}";;
            esac])

MU_CHECK_TLS
if test "$WITH_GNUTLS" != "no"; then
  MU_AUTHLIBS="$MU_AUTHLIBS $TLS_LIBS"
  AC_DEFINE(WITH_GNUTLS,1,[Define if mailutils is using GNU TLS])
  AC_DEFINE(WITH_TLS,1,[Define if mailutils is using TLS])
fi

AC_SUBST(SITE_MAIL_RC)
AC_ARG_WITH([mail-rc],
            AC_HELP_STRING([--with-mail-rc=FILE],
                           [use FILE instead of $sysconfdir/mail.rc]),
            [
case "${withval}" in
  /*) SITE_MAIL_RC="${withval}";;
  *)  SITE_MAIL_RC="\$(sysconfdir)/${withval}";;
  esac],
 [SITE_MAIL_RC="\$(sysconfdir)/mail.rc"])

MU_CONF_MAILDIR=
AC_ARG_WITH([mail-spool],
            AC_HELP_STRING([--with-mail-spool=PATH],
                           [use PATH instead of /var/spool/mail]),
            [
case "$withval" in
  /*/) MU_CONF_MAILDIR="\"${withval}\"";;
  /*)  MU_CONF_MAILDIR="\"${withval}/\"";;
  *:*[[/=]]) MU_CONF_MAILDIR="\"${withval}\"";;
  *)  AC_MSG_ERROR(--with-mail-spool must be an absolute path or a valid URL);;
esac])

if test "x$MU_CONF_MAILDIR" != x; then
  AC_DEFINE_UNQUOTED(MU_CONF_MAILDIR,$MU_CONF_MAILDIR,
                     [Pathname of the mail spool directory])
fi

PATHDEFS_H=pathdefs.h
MU_CONFIG_PATHS($srcdir/paths,PATHDEFS_H)

AH_BOTTOM(
[/* Defaults for PATH_ variables.  */
#include <confpaths.h>

#ifdef MU_CONF_MAILDIR
# define MU_PATH_MAILDIR MU_CONF_MAILDIR
#else
# define MU_PATH_MAILDIR PATH_MAILDIR "/"
#endif])

use_dbm=no
AC_ARG_WITH([gdbm],
            AC_HELP_STRING([--with-gdbm],
                           [use GNU DBM]),
            [
case "${withval}" in
  yes) use_dbm=GDBM ;;
  no)  use_dbm=no ;;
  *)   AC_MSG_ERROR(bad value ${withval} for --with-gdbm) ;;
esac])

## Support --with-db2 for backward compatibility
if test "${with_db2+set}" = set; then
  case "${with_db2}" in
  yes) use_dbm=BDB2 ;;
  no)  use_dbm=no ;;
  *)   AC_MSG_ERROR(bad value ${with_db2} for --with-db2) ;;
  esac
fi

AC_ARG_WITH([berkeley-db],
            AC_HELP_STRING([--with-berkeley-db],
                           [use Berkeley DB]),
            [
case "${withval}" in
  yes) use_dbm=BDB ;;
  no)  use_dbm=no ;;
  *)   use_dbm=BDB="${withval}";;
esac])

AC_ARG_WITH([ndbm],
            AC_HELP_STRING([--with-ndbm],
                           [use NDBM]),
            [
case "${withval}" in
  yes) use_dbm=NDBM ;;
  no)  use_dbm=no ;;
  *)   AC_MSG_ERROR(bad value ${withval} for --with-ndbm) ;;
esac])

AC_ARG_WITH([tokyocabinet],
            AC_HELP_STRING([--with-tokyocabinet],
                           [use Tokyo Cabinet]),
            [
case "${withval}" in
  yes) use_dbm=TC ;;
  no)  use_dbm=no ;;
  *)   AC_MSG_ERROR(bad value ${withval} for --with-tokyocabinet) ;;
esac])

AC_MSG_CHECKING(for log facility)
log_facility="LOG_MAIL"
AC_ARG_WITH([log-facility],
            AC_HELP_STRING([--with-log-facility=facility],
                           [enable logging to the given facility]),
            [AC_TRY_COMPILE([#include <syslog.h>], int lf = $withval,
	     log_facility=$withval)])
AC_DEFINE_UNQUOTED(LOG_FACILITY, $log_facility,
                   [Default syslog facility])
fstr=`echo "$log_facility" | tr A-Z a-z | sed 's/log_//'`
AC_DEFINE_UNQUOTED(LOG_FACILITY_STRING, "$fstr",
                   [String representation of the default syslog facility])
		   
AC_MSG_RESULT($log_facility)

dnl Check for headers
AC_HEADER_STDC
AC_HEADER_DIRENT
AC_CHECK_HEADERS(errno.h fcntl.h inttypes.h libgen.h limits.h\
 malloc.h obstack.h paths.h shadow.h socket.h sys/socket.h stdarg.h stdio.h\
 stdlib.h string.h strings.h sys/file.h sysexits.h syslog.h termcap.h\
 termios.h termio.h sgtty.h utmp.h utmpx.h unistd.h wchar.h)

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_UID_T
AC_TYPE_MODE_T
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPE(ssize_t, , AC_DEFINE(ssize_t, int, [Define to int if <sys/types.h> does not define]))
AC_TYPE_SIGNAL
AC_DECL_SYS_SIGLIST
AC_STRUCT_TIMEZONE
AC_CHECK_TYPE(ino_t, , AC_DEFINE(ino_t, unsigned long, [Define to unsigned long if <sys/types.h> does not define]))
AC_CHECK_TYPE(dev_t, , AC_DEFINE(dev_t, unsigned long, [Define to unsigned long if <sys/types.h> does not define]))
AC_CHECK_TYPE(socklen_t, , AC_DEFINE(socklen_t, int, [Define to int if <sys/types.h> does not define]),
[
#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
# include <sys/socket.h>
#endif
#ifdef HAVE_SOCKET_H
# include <socket.h>
#endif
])
AC_CHECK_TYPE(long long, , AC_DEFINE(LONG_LONG, long, [Define to long if compiler does not support]))
AC_CHECK_TYPE(long double, , AC_DEFINE(LONG_DOUBLE, double, [Define to double if compiler does not support]))

AC_SUBST(MU_ULONG_MAX)
AC_RUN_IFELSE([AC_LANG_PROGRAM(
[#include <stdio.h>
#include <stdlib.h>
#include <limits.h>],
[
 FILE *f = fopen ("conftest.val", "w");
 if (! f)
    exit (1);
 fprintf(f, "%lu\n", ULONG_MAX);
 fclose(f);
 return 0;]
)],
 [MU_ULONG_MAX=`cat conftest.val`],
 [MU_ULONG_MAX=0],
 [MU_ULONG_MAX=0])

AC_STRUCT_TM
AC_CHECK_MEMBERS([struct tm.tm_zone, struct tm.tm_gmtoff, struct tm.tm_isdst],
                 ,,
                 [#include <sys/types.h>
#include <$ac_cv_struct_tm>])

dnl Check for working functions

#####
# Determine the size of off_t and define an equivalent mu_off_t type.
# We cannot use off_t in the library since its size depends on whether
# large file support is used or not, which will lead to arbitrary crashes
# if a user application and the library use off_t of different sizes.

AC_CHECK_SIZEOF(off_t)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(long long)
AC_SUBST(MU_OFF_TYPE)
if test $ac_cv_sizeof_off_t -eq $ac_cv_sizeof_long; then
  MU_OFF_TYPE=long
elif test $ac_cv_sizeof_off_t -eq $ac_cv_sizeof_long_long; then
  MU_OFF_TYPE="long long"
else
  AC_MSG_ERROR([Cannot find ${ac_cv_sizeof_off_t}-byte type, suitable for mu_off_t])
fi
AC_FUNC_MMAP

AC_CHECK_DECLS([strtok_r, strchrnul, strndup, asprintf, vasprintf, strsignal],
 , , [
#include <stdio.h>
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif])

AC_CHECK_DECLS(environ, , ,
  [ #include <unistd.h>
    #include <stdlib.h> ])

AH_BOTTOM([
#if !HAVE_DECL_STRTOK_R
extern char *strtok_r (char *s, const char *delim, char **save_ptr);
#endif
#if !HAVE_DECL_STRSIGNAL
extern char *strsignal (int);
#endif
])

AC_CHECK_FUNCS(mkstemp sigaction sysconf getdelim setreuid \
 setresuid seteuid setlocale vfork _exit)

AC_FUNC_FSEEKO
AC_FUNC_SETVBUF_REVERSED

AH_BOTTOM([
#if !defined (HAVE_FSEEKO)
# define fseeko fseek
#endif
#if defined (SETVBUF_REVERSED)
# define SETVBUF(str,buf,mode,size) setvbuf(str,mode,buf,size)
#else
# define SETVBUF(str,buf,mode,size) setvbuf(str,buf,mode,size)
#endif
])

AC_CHECK_FUNCS(fgetpwent)

## Check for presence of utmp.h and utmp-related functions
MU_CHECK_UTMP

dnl Check for libraries

AH_BOTTOM([
/* program_name is used by lib/error.c */
#define program_name program_invocation_name
])


## FriBidi support

AC_CHECK_FUNCS(wcwidth)

AC_SUBST(FRIBIDI_LIBS)

use_fribidi=yes

AC_ARG_WITH([fribidi],
            AC_HELP_STRING([--without-fribidi],
                           [do not compile FriBidi support]),
            [use_fribidi=$withval])

AH_TEMPLATE(HAVE_LIBFRIBIDI,
            [Define to 1 if you have the `fribidi' library (-lfribidi).])
AH_TEMPLATE(HAVE_FRIBIDI_CHARSET_CAP_RTL,
            [Define to 1 if fribidi headers declare FRIBIDI_CHARSET_CAP_RTL type])
AH_BOTTOM([
#ifndef HAVE_FRIBIDI_CHARSET_CAP_RTL
# define FRIBIDI_CHARSET_CAP_RTL FRIBIDI_CHAR_SET_CAP_RTL
#endif])

if test "$use_fribidi" = "yes"; then
  AC_CHECK_HEADERS(fribidi/fribidi.h)
  if test $ac_cv_header_fribidi_fribidi_h = yes; then
    AC_CHECK_LIB(fribidi, fribidi_log2vis,
                 [AC_DEFINE(HAVE_LIBFRIBIDI,1)
                  FRIBIDI_LIBS=-lfribidi],
                 [use_fribidi=no])

    AC_RUN_IFELSE(AC_LANG_PROGRAM([#include <fribidi/fribidi.h>],
                                  [FriBidiCharSet s = FRIBIDI_CHARSET_CAP_RTL;]),
                  [AC_DEFINE(HAVE_FRIBIDI_CHARSET_CAP_RTL)])
                 
  else
    use_fribidi=no
  fi
fi


## SQL support

AH_TEMPLATE(USE_SQL,[Enables SQL support])
AH_TEMPLATE(HAVE_MYSQL,[Define if you have libmysql])
AH_TEMPLATE(HAVE_PGSQL,[Define if you have libpq])

SQL_MODULES=
AC_SUBST(SQL_MODULES)

SQL_LIBOBJS=
AC_SUBST(SQL_LIBOBJS)

SQL_LTLIBOBJS=
AC_SUBST(SQL_LTLIBOBJS)

BUILD_SQL=
AC_SUBST(BUILD_SQL)
AC_SUBST(SQLLIB)
AC_SUBST(SQLLIB_DEPENDENCY)

SQL_LOADABLE_MODULES=
AC_SUBST(SQL_LOADABLE_MODULES)

AC_DEFUN([MU_SQL_LDADD],[
  if test "$USE_LOADABLE_MODULES" = yes; then
    $1=$2
  else
    SQLLIB="$SQLLIB $2" 
  fi 			
])

add_sql_module() {
	case $SQL_MODULES in
	*$1*) AC_MSG_FAILURE([$1 specified twice]);;
	*)    SQL_MODULES="$SQL_MODULES $1";;
	esac
}

## *********
## MySQL
## *********
AC_SUBST(MYSQLLIBS)

mysql_init() {
  MU_CHECK_LIB(mysqlclient, 
	       mysql_real_connect, 
	       [-lm],
               [ AC_DEFINE(USE_SQL)
                 AC_DEFINE(HAVE_MYSQL)
	         MU_SQL_LDADD(MYSQLLIBS, $mu_cv_lib_mysqlclient)
                 status_mysql=yes ],
	       [ AC_MSG_FAILURE([The requested library libmysqlclient is not found or is unusable])],
               [/usr/local/lib/mysql /usr/lib/mysql])
}

AC_ARG_WITH(mysql,
	AC_HELP_STRING([--with-mysql],
	[Configure to work with MySQL]),
	[case $withval in
	 yes)   add_sql_module "mysql"
		;;
	 no)    ;;
        esac])


# *********
# Postgres
# *********

postgres_init() {
  MU_CHECK_LIB(pq, PQconnectStart, [],
	       [ AC_DEFINE(USE_SQL)
                 AC_DEFINE(HAVE_PGSQL)
	         MU_SQL_LDADD(POSTGRESLIBS, $mu_cv_lib_pq)
                 status_pgsql=yes ],
	       [ AC_MSG_FAILURE([The requested library libpq is not found or is unusable])],
               [/usr/local/pgsql/lib /usr/pgsql/lib])
}

AC_ARG_WITH(postgres,
	AC_HELP_STRING([--with-postgres],
	[Configure to work with Postgres]),
	[case $withval in
	 yes)	add_sql_module "postgres"
		;;
	 no)    ;;
        esac])


# *********
# ODBC
# *********
AH_TEMPLATE(USE_SQL_ODBC,[When defined, enable ODBC support])
AH_TEMPLATE(HAVE_LIBODBC,[Define if you have libodbc or libiodbc]) 
AC_SUBST(ODBCLIBS)

check_lib_odbc() {
  MU_CHECK_LIB(odbc, SQLAllocHandle, [],
	       [ AC_DEFINE(USE_SQL_ODBC)
                 AC_DEFINE(HAVE_LIBODBC)
	         MU_SQL_LDADD(ODBCLIBS, $mu_cv_lib_odbc)],
               [AC_MSG_FAILURE([The requested library libobdc is not found or is unusable])],
               [/usr/local/lib])
}

check_lib_iodbc() {
  MU_CHECK_LIB(iodbc, SQLAllocHandle, [],
	       [ AC_DEFINE(USE_SQL_ODBC)
                 AC_DEFINE(HAVE_LIBODBC)
		 MU_SQL_LDADD(ODBCLIBS, $mu_cv_lib_iodbc)],
               [AC_MSG_FAILURE([The requested library libiodbc is not found or is unusable])],
               [/usr/local/lib])
}

with_flag_used=odbc

odbc_init() {
         case $with_flag_used in
	 yes|odbc|ODBC)
		check_lib_odbc
		;;
	 iodbc|iODBC|IODBC)
		check_lib_iodbc
		;;
	 no)	;;
	 esac
}

AC_ARG_WITH(odbc,
	AC_HELP_STRING([--with-odbc[={odbc|iodbc}]],
	[Configure to work with ODBC]),
	[case $withval in
	 yes|odbc|ODBC|iodbc|iODBC|IODBC)
		add_sql_module "odbc"
		with_flag_used=$withval
		;;
	 no)    with_flag_used=$withval
		;;
	 *)	AC_MSG_ERROR([unsupported option value: $withval])
        esac])


### General SQL support
AC_ARG_WITH(sql,
        AC_HELP_STRING([--with-sql=modlist],
                       [Configure to work with SQL]),
	[for mod in `echo $withval | tr ':,' '  '`
         do
		if test -r $srcdir/sql/${mod}.c; then
			add_sql_module "${mod}"
		else
			AC_MSG_ERROR(No such SQL module: $mod)
		fi
	 done])

dummy_init() {
	:
}

if test -n "$SQL_MODULES"; then
  USE_SQL=1
  AC_DEFINE(USE_SQL)
  BUILD_SQL=libsql.la
  SQLLIB=../sql/libsql.la
  SQLLIB_DEPENDENCY=$SQLLIB

  for module in $SQL_MODULES
  do
    ${module:-dummy}_init
    if test $USE_LOADABLE_MODULES = yes; then
      SQL_LOADABLE_MODULES="$SQL_LOADABLE_MODULES ${module}.la"
    else
      SQL_LIBOBJS="$SQL_LIBOBJS ${module}.o"
    fi
  done
  S=`echo "$SQL_LIBOBJS" |
                  sed 's,\.[[^.]]* ,$U&,g;s,\.[[^.]]*$,$U&,'`
  SQL_LTLIBOBJS=`echo "$S" |
                  sed 's,\.[[^.]]* ,.lo ,g;s,\.[[^.]]*$,.lo,'`

  AC_CONFIG_COMMANDS([sql/modlist.h],
                     [$srcdir/mu-aux/sqlmod.sh $SQL_MODULES > sql/modlist.h],
                     [if test $USE_LOADABLE_MODULES = yes; then
			SQL_MODULES=
                      else
                        SQL_MODULES="$SQL_MODULES"
                      fi])
fi


# LDAP support
AC_ARG_WITH(ldap,
	AC_HELP_STRING([--with-ldap],
	[Configure to work with OpenLDAP]),
	[case $withval in
	 yes)	status_ldap=maybe
		;;
	 no)    status_ldap=no
                ;;
        esac],
        [status_ldap=maybe])

if test $status_ldap = maybe; then
  AC_CHECK_LIB(ldap, ldap_bind,[
   status_ldap=yes
   AC_DEFINE(WITH_LDAP, 1, [Define to 1 if you want to use -lldap])
   MU_AUTHLIBS="$MU_AUTHLIBS -lldap"],
   [status_ldap=no])
fi


# Virtual domain support
MU_ENABLE_SUPPORT(virtual-domains)
AC_SUBST(SITE_VIRTUAL_PWDDIR)
AC_ARG_WITH([virtual-pwddir],
             AC_HELP_STRING([--with-virtual-pwddir=DIR],
                              [use DIR instead of $sysconfdir/domain]),[
             case "${withval}" in
		/*) SITE_VIRTUAL_PWDDIR="${withval}";;
		*)  SITE_VIRTUAL_PWDDIR="\$(sysconfdir)/${withval}";;
	     esac],
             [SITE_VIRTUAL_PWDDIR="\$(sysconfdir)/domain"])
MU_ENABLE_SUPPORT(imap)
MU_ENABLE_SUPPORT(pop)
MU_ENABLE_SUPPORT(nntp)

AC_SUBST(MU_SMTP_PROGRAMS_BUILD)
AC_SUBST(MU_SMTP_DEJATOOL)

MU_ENABLE_SUPPORT(smtp,
                  [MU_SMTP_PROGRAMS_BUILD='$(MU_SMTP_PROGRAMS_LIST)'
                   MU_SMTP_DEJATOOL='$(SMTP_DEJATOOL)'])
MU_ENABLE_SUPPORT(sendmail)

MU_ENABLE_SUPPORT(prog)

MU_ENABLE_SUPPORT(mh)

MU_ENABLE_SUPPORT(maildir)

# FIXME: Should be --with-radius 
MU_ENABLE_SUPPORT(radius,
                  [AM_GNU_RADIUS(1.6,
		                 [MU_AUTHLIBS="$MU_AUTHLIBS -lgnuradius"
				  mu_cv_enable_radius=yes],
				 [mu_cv_enable_radius=no])]) 

AC_ARG_ENABLE([experimental],
              AC_HELP_STRING([--enable-experimental],
                             [build experimental and/or unfinished utilities]),
              [mu_build_experimental="${enableval}"],
              [mu_build_experimental=no])

if test x"$testpam" = x"yes"; then
	AC_CHECK_HEADERS(security/pam_appl.h)
	if test x"$ac_cv_header_security_pam_appl_h" = x"yes"; then
		AC_CHECK_LIB(dl, dlopen, [LDL=-ldl], [LDL=])
		AC_CHECK_LIB(pam, pam_start,
			[MU_AUTHLIBS="$MU_AUTHLIBS -lpam $LDL"
                         ac_cv_have_libpam="yes"],, $LDL)
		AC_DEFINE(USE_LIBPAM,1,[Enable use of PAM])
		status_pam=yes
	fi
fi

AC_CHECK_HEADERS(crypt.h)
AC_CHECK_DECLS(crypt, , ,[
#include <unistd.h>
#ifdef HAVE_CRYPT_H
# include <crypt.h>
#endif])
AH_BOTTOM([
#if !HAVE_DECL_CRYPT
char *crypt(const char *key, const char *salt);
#endif])

AC_CHECK_LIB(crypt, crypt)

#When using thread support some platforms need -D_REENTRANT to get the
#right prototypes including errno.
dnl Check threading support
# We have to rearrange things a little, it appears that the new autoconf
# does not like long cascading AC_CHECK_LIB.
if test x"$usepthread" = x"yes"; then
	AC_CHECK_LIB(pthread, pthread_cancel, have_libpthread=yes,
		 have_libpthread=no)
        if test x"$have_libpthread" = x"yes"; then
		AC_CHECK_HEADERS(pthread.h,
                                 AC_DEFINE(WITH_PTHREAD,1,
                                           [Define this to enable POSIX threads safety]))
			LIBS="$LIBS -lpthread"
			AC_DEFINE(_REENTRANT,1,[Reentrant calls])
			AC_CHECK_FUNCS(pthread_rwlock_init)
	else
		AC_SEARCH_LIBS(pthread_cancel, pthread,
			[AC_CHECK_FUNCS(pthread_rwlock_init)
				AC_CHECK_HEADERS(pthread.h,
					AC_DEFINE(WITH_PTHREAD,1,
                                                  [Define this to enable POSIX threads safety]))
					AC_DEFINE(_REENTRANT,1,[Reentrant calls])])
	fi
fi

dnl Do we need extra libs for networking?
AC_CHECK_FUNC(gethostbyname, [true],
	AC_CHECK_LIB(nsl, gethostbyaddr, LIBS="-lnsl $LIBS"))

AC_CHECK_FUNC(socket, [true],
	AC_CHECK_LIB(socket, socket, LIBS="-lsocket $LIBS"))

dnl Check for Curses libs.
for lib in ncurses curses termcap
do
	AC_CHECK_LIB($lib, tputs, [CURSES_LIBS="-l$lib"; break])
done

AC_SUBST(CURSES_LIBS)

dnl Check for GNU Readline
AC_SUBST(READLINE_LIBS)

if test x"$usereadline" = x"yes"; then

	dnl FIXME This should only link in the curses libraries if it's
	dnl really needed!

	saved_LIBS=$LIBS
	LIBS="$LIBS $CURSES_LIBS"
	AC_CHECK_LIB(readline, readline, mu_have_readline=yes)
	LIBS=$saved_LIBS

	if test x"$mu_have_readline" = x"yes"; then
		AC_CHECK_HEADERS(readline/readline.h,
			AC_DEFINE(WITH_READLINE,1,[Enable use of readline]))
			READLINE_LIBS="-lreadline $CURSES_LIBS"
			saved_LIBS=$LIBS
			LIBS="$LIBS $READLINE_LIBS"
			AC_CHECK_FUNCS(rl_completion_matches)
			LIBS=$saved_LIBS
	else
		AC_MSG_WARN(readline requested but does not seem to be installed)
	fi

fi

AH_BOTTOM([
/* Newer versions of readline have rl_completion_matches */
#ifndef HAVE_RL_COMPLETION_MATCHES
# define rl_completion_matches completion_matches
#endif])

dnl Check for DBM

AH_TEMPLATE([WITH_BDB],
            [Define to the major version of Berkeley DB library to use])

## Set the variable status_dbm to $1 if:
##
##  1. Function $3 is defined in the library $2
##  2. Header file db.h is available
##
## Then check if the major version, minor version and patchlevel of the
## library matches those from the header. If so, define WITH_BDB
## to the version (i.e. $1 with all dots removed). Otherwise, report
## an error and stop.
##
check_bdb() {
        ver=`echo $1 | tr -d '.'`
	major=`expr $ver : '\(.\).*'`
	AC_CHECK_LIB($2, $3,
                     [AC_CHECK_HEADERS(db.h)
                      if test $ac_cv_header_db_h = yes; then
                        LIBS="$LIBS -l$2"
                        MU_DB2_CURSOR
                        status_dbm=$1
                      fi])
	if test "$status_dbm" = no; then
		:
	else
	  AC_RUN_IFELSE(
           [AC_LANG_PROGRAM([#include "db.h"],
            [int v_major, v_minor, v_patch;
             db_version(&v_major, &v_minor, &v_patch);
             return !(DB_VERSION_MAJOR == $major
                       && v_major == DB_VERSION_MAJOR
                       && v_minor == DB_VERSION_MINOR
                       && v_patch == DB_VERSION_PATCH);
            ])],
           [],
           [status_dbm=no])
           if test "$status_dbm" != no; then
	     AC_DEFINE_UNQUOTED(WITH_BDB,$ver)
           fi
	fi
}

## Check for the Berkeley DB library version $1, assuming Slackware-like
## installation layout (header files in /usr/incude/db$vn and library named
## libdb-$version.so, where $version is the library version and $vn is
## $version with all dots removed.
##
check_slackware_bdb() {
	dir=db`echo $1|tr -d '.'`
	save_CPPFLAGS=$CPPFLAGS
	CPPFLAGS="$CPPFLAGS -I/usr/include/$dir"
	check_bdb "$1" db-$1 db_create
	CPPFLAGS=$save_CPPFLAGS
	if test "$status_dbm" = "$1"; then
	    MU_COMMON_INCLUDES="$MU_COMMON_INCLUDES -I/usr/include/$dir"
	fi
}

case "$use_dbm" in
GDBM)
	AC_CHECK_LIB(gdbm, gdbm_open,
                     [AC_CHECK_HEADERS(gdbm.h,
                                       AC_DEFINE(WITH_GDBM,1,
                                                 [Enable use of GNU DBM library]))
                     LIBS="$LIBS -lgdbm"
                     status_dbm="GDBM"]);;

BDB2)   check_bdb 2 db db_open
        test -n "$status_dbm" && status_dbm="Berkeley DB v. $status_dbm";;

BDB)    for version in 4 3 2
	do
		case $version in
		4|3) func=db_create;;
		2)   func=db_open;;
		esac
		check_bdb $version db $func
		if test "$status_dbm" != no; then
                   status_dbm="Berkeley DB v. $status_dbm"
		   break;
		fi
	done;;

BDB=*)
	name=`expr $use_dbm : 'BDB=\(.*\)'`
	case $name in
	[[0-9]]*)  check_slackware_bdb $name;;
	*)	   for version in 4 3 2
	           do
			case $version in
			4|3) func=db_create;;
			2)   func=db_open;;
			esac
			check_bdb $version $name $func
			if test "$status_dbm" != no; then
		           status_dbm="Berkeley DB v. $status_dbm"
			   break;
			fi
		   done
		   ;;
	esac
        ;;

NDBM)
	AC_CHECK_LIB(ndbm, dbm_open,
                     [AC_CHECK_HEADERS(ndbm.h,
                                       AC_DEFINE(WITH_NDBM,1,
                                                 [Enable use of NDBM]))
                     LIBS="$LIBS -lndbm"
                     status_dbm="NDBM"]);;

TC)
	AC_CHECK_LIB(tokyocabinet, tchdbnew,
                     [AC_CHECK_HEADERS(tchdb.h,
                                       AC_DEFINE(WITH_TOKYOCABINET,1,
                                                 [Enable use of Tokyo Cabinet]))
                     LIBS="$LIBS -ltokyocabinet -lz -lbz2 -lrt"
                     status_dbm="Tokyo Cabinet"]);;
esac

AC_SUBST(POPAUTH)
if test "$status_dbm" != no; then
  POPAUTH='popauth$(EXEEXT)'
fi

dnl Check for Guile
AC_SUBST([GUILE_BINDIR])
AC_SUBST([LIBMU_SCM])
AC_SUBST([LIBMU_SCM_DEPS])
AC_SUBST([MU_GUILE_SIEVE_MOD_DIR])
AC_SUBST([MU_GUILE_SIEVE_MOD_DATA])
GINT_INIT([gint],[1.8 with-guile],
	  [useguile=yes
 	   AC_DEFINE([WITH_GUILE],1,[Enable Guile support])
           GUILE_BINDIR=`guile-config info bindir`
           LIBMU_SCM=../libmu_scm/libmu_scm.la
           LIBMU_SCM_DEPS='${MU_LIB_MBOX} ${MU_LIB_IMAP} ${MU_LIB_POP} ${MU_LIB_MH} ${MU_LIB_MAILDIR} ${MU_LIB_MAILER}'
           MU_GUILE_SIEVE_MOD_DIR='$(GUILE_SITE)/$(PACKAGE)/sieve-modules'
           MU_GUILE_SIEVE_MOD_DATA='$(MU_GUILE_SIEVE_MOD_DATA_X)'
           GINT_INCLUDES='${MU_APP_COMMON_INCLUDES}'
           GINT_LDADD=../lib/libmuaux.la
          ],[useguile=no])

AM_CONDITIONAL([MU_COND_LIBMU_SCM],[test "$useguile" = "yes"])

dnl Check for Emacs site-lisp directory
AM_PATH_LISPDIR

if test "$EMACS" != "no"; then
	lisp_LISP='$(LISPSRC)'
fi
AC_SUBST(lisp_LISP)

# Check for Python
AC_ARG_WITH([python],
            AC_HELP_STRING([--without-python],
                           [do not build Python interface]),
            [
case "${withval}" in
  yes) status_python=yes ;;
  no)  status_python=no ;;
  *)   AC_MSG_ERROR(bad value ${withval} for --without-python) ;;
esac],[status_python=yes])

AC_SUBST(PYTHON_LIBS)
AC_SUBST(PYTHON_INCLUDES)

if test "$status_python" = yes; then
  AM_PATH_PYTHON(2.5.0,, [status_python=no])
  if test "$status_python" = yes; then
    AC_ARG_VAR([PYTHON_CONFIG], [The name of python-config binary])
    AC_PATH_PROG([PYTHON_CONFIG], python-config)
    if test -n "$PYTHON_CONFIG"; then
      PYTHON_LIBS=`python-config --libs`
      PYTHON_INCLUDES=`python-config --includes`
    else
      status_python=no     
    fi

    if test "$status_python" = yes; then
      AC_DEFINE(WITH_PYTHON,1,[Enable Python support])
      MU_LIB_PY='${top_builddir}/python/libmu_py/libmu_py.la'
      AC_SUBST([PYTHON_SITE_DIR])
      AC_SUBST([PYTHON_EXEC_DIR])
      AC_ARG_WITH([pythondir],
           AC_HELP_STRING([--with-pythondir=DIR],
                          [Specify the directory to install Python modules to]),
           [case $withval in
            /*)  PYTHON_SITE_DIR="${withval}/mailutils"
	         PYTHON_EXEC_DIR="${withval}/mailutils";;
            *)   AC_MSG_ERROR([Argument to --with-pythondir must be an absolute directory name]);;
            esac],
	   [PYTHON_SITE_DIR='$(pkgpythondir)'
	    PYTHON_EXEC_DIR='$(pkgpyexecdir)'])
    fi
  fi
fi
AM_CONDITIONAL([MU_COND_PYTHON], [test "$status_python" = yes])

AC_SUBST(MU_CXX_EXAMPLES_BUILD)
MU_ENABLE_SUPPORT(cxx,
  	          [MU_CXX_EXAMPLES_BUILD='$(MU_CXX_EXAMPLES_LIST)'])
AM_CONDITIONAL([MU_COND_LIBMU_CPP], [test "$mu_cv_enable_cxx" = yes])

# Default mailbox record
# Note: 1. Support for mbox type is always enabled.
#       2. Only local mailbox types are allowed for MU_DEFAULT_SCHEME
AC_ARG_VAR([MU_DEFAULT_SCHEME],
           [Default mailbox record. Allowed values are: mbox (default), mh, and maildir.])
if test -z "$MU_DEFAULT_SCHEME"; then
  MU_DEFAULT_SCHEME=mbox
fi  
case $MU_DEFAULT_SCHEME in
mbox)
   ;;
mh|maildir)
   eval testval=\$mu_cv_enable_$MU_DEFAULT_SCHEME
   if test "$testval" != yes; then
     AC_MSG_ERROR([Cannot set default mailbox record type: support for $MU_DEFAULT_SCHEME is disabled])
   fi
   ;;
*) AC_MSG_ERROR([Unknown or not allowed mailbox scheme: $MU_DEFAULT_SCHEME]);;
esac
AC_DEFINE_UNQUOTED(MU_DEFAULT_SCHEME, "$MU_DEFAULT_SCHEME",
                   [Default mailbox scheme.])
AC_SUBST(MU_DEFAULT_RECORD,mu_${MU_DEFAULT_SCHEME}_record)

AC_ARG_VAR([DEFAULT_CUPS_CONFDIR],
           [Set the location of CUPS configuration directory. Default is \$sysconfdir/cups])

AC_SUBST(DEFAULT_CUPS_CONFDIR)
if test -z "$DEFAULT_CUPS_CONFDIR"; then
  DEFAULT_CUPS_CONFDIR='$(sysconfdir)/cups'
fi

dnl ***********************************************************************
dnl Configure particular utilities
dnl ***********************************************************************

MU_ENABLE_BUILD(frm)
MU_ENABLE_BUILD(pop3d)
MU_ENABLE_BUILD(imap4d)
MU_ENABLE_BUILD(comsat)
MU_ENABLE_BUILD(maidag)
MU_ENABLE_BUILD(mail)
MU_ENABLE_BUILD(sieve)
MU_ENABLE_BUILD(guimb,,,[test "$useguile" = yes])
MU_ENABLE_BUILD(messages)
MU_ENABLE_BUILD(readmsg)
MU_ENABLE_BUILD(dotlock)
MU_ENABLE_BUILD(movemail)
MU_ENABLE_BUILD(mimeview)

MU_ENABLE_BUILD(mh,,,[test "$mu_cv_enable_mh" = yes])

AC_SUBST(MH_BIN_DIR)

AC_ARG_WITH([mh-bindir],
	    AC_HELP_STRING([--with-mh-bindir],
                           [Set the directory where to install the MH binaries. If it starts with '/' it is taken as an absolute path specification, otherwise ${prefix} is prepended to it. (Default: ${exec_prefix}/bin/mu-mh)]),
		    [case "${withval}" in
		     /*)  MH_BIN_DIR=${withval};;
                     *)   MH_BIN_DIR="\${prefix}/${withval}";;
                     esac],
                    [MH_BIN_DIR='${exec_prefix}/bin/mu-mh'])



dnl Make sysconfdir available to the application
dnl This must be done LAST, since CPPFLAGS is passed by configure
dnl to build tests, but this relies on being run in make to
dnl get sysconfdir expanded.

CPPFLAGS="$CPPFLAGS -DSYSCONFDIR=\\\"\$(sysconfdir)\\\""

# Doc hints.
# Select a rendition level:
#  DISTRIB for stable releases (at most one dot in the version number)
#  and maintenance releases (two dots, patchlevel < 50)
#  PROOF for alpha releases.
#  PUBLISH can only be required manually when running make in doc/
AC_SUBST(RENDITION)
case `echo $VERSION|sed  's/[[^.]]//g'` in
""|".")  RENDITION=DISTRIB;;
"..")  if test `echo $VERSION | sed  's/.*\.//'` -lt 50; then
	 RENDITION=DISTRIB
       else
         RENDITION=PROOF
       fi;;
*)     RENDITION=PROOF;;
esac


AC_CONFIG_COMMANDS([status],[
cat <<EOF

*******************************************************************
GNU Mailutils configured with the following settings:

Default mailbox scheme ........ $status_scheme
Use PAM ....................... $status_pam
Use -ltdl ..................... $status_ltdl
Use DBM ....................... $status_dbm
Use GNU TLS ................... $status_gnutls
Use GSASL ..................... $status_gsasl
Use GSSAPI .................... $status_gssapi
Use TCP wrappers .............. $status_tcpwrap
Pthread support ............... $status_pthread
Readline support .............. $status_readline
MySQL support ................. $status_mysql
PostgreSQL support ............ $status_pgsql
LDAP support .................. $status_ldap
Radius support ................ $status_radius
Support for virtual domains ... $status_virtual_domains

Interfaces:

Guile ......................... $status_guile
C++ ........................... $status_cxx
Python ........................ $status_python

Mailbox formats:

IMAP .......................... $status_imap
POP ........................... $status_pop
MH ............................ $status_mh
maildir ....................... $status_maildir

Mailers:

SMTP .......................... $status_smtp
Sendmail ...................... $status_sendmail

*******************************************************************

Before proceeding, verify if these satisfy your requirements. 
EOF
],
[status_scheme=$MU_DEFAULT_SCHEME
status_pam=$status_pam
status_ltdl=$status_ltdl
status_dbm="$status_dbm"
status_ldap=$status_ldap
status_gnutls=$WITH_GNUTLS
status_gsasl=$status_gsasl
status_gssapi=$WITH_GSSAPI
status_guile=$useguile
status_python=$status_python
status_cxx=$mu_cv_enable_cxx
status_tcpwrap=$status_tcpwrap
status_pthread=$usepthread
status_readline=$usereadline
status_mysql=$status_mysql
status_pgsql=$status_pgsql
status_radius=$mu_cv_enable_radius
status_virtual_domains=$mu_cv_enable_virtual_domains
status_imap=$mu_cv_enable_imap
status_pop=$mu_cv_enable_pop
status_mh=$mu_cv_enable_mh
status_maildir=$mu_cv_enable_maildir
status_smtp=$mu_cv_enable_smtp
status_sendmail=$mu_cv_enable_sendmail
])

dnl Output Makefiles
AC_CONFIG_FILES([
 include/confpaths.h
 Makefile
 sql/Makefile
 comsat/Makefile
 config/Makefile
 doc/Makefile
 doc/man/Makefile
 doc/texinfo/Makefile
 dotlock/Makefile
 examples/Makefile
 examples/config/Makefile
 examples/cpp/Makefile
 examples/python/Makefile
 examples/scheme/Makefile
 gint/Makefile
 frm/Makefile
 frm/testsuite/Makefile
 guimb/Makefile
 guimb/scm/Makefile
 imap4d/Makefile
 imap4d/testsuite/Makefile
 include/Makefile
 include/mailutils/Makefile
 include/mailutils/sys/Makefile
 include/mailutils/cpp/Makefile
 lib/Makefile
 libmu_argp/Makefile
 libmu_auth/Makefile
 libmu_cfg/Makefile
 libmu_cpp/Makefile
 libmu_scm/Makefile
 libmu_sieve/Makefile
 libmu_sieve/extensions/Makefile
 libproto/Makefile
 libproto/mbox/Makefile
 libproto/mh/Makefile
 libproto/maildir/Makefile
 libproto/mailer/Makefile
 libproto/pop/Makefile
 libproto/nntp/Makefile
 libproto/imap/Makefile
 libproto/include/Makefile
 maidag/Makefile
 mail/Makefile
 mail/testsuite/Makefile
 mailbox/Makefile
 mailbox/testsuite/Makefile
 messages/Makefile
 messages/testsuite/Makefile
 mh/Makefile
 mimeview/Makefile
 movemail/Makefile
 po/Makefile.in
 pop3d/Makefile
 pop3d/testsuite/Makefile
 python/Makefile
 python/libmu_py/Makefile
 python/mailutils/Makefile
 readmsg/Makefile
 readmsg/testsuite/Makefile
 mu-aux/Makefile
 mu-aux/mailutils.spec
 sieve/Makefile
 sieve/testsuite/Makefile
 testsuite/Makefile
 ])
AC_OUTPUT

