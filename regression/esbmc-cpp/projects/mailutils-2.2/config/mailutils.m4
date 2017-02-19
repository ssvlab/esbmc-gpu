dnl Copyright (C) 2006, 2007, 2010 Free Software Foundation, Inc.
dnl
dnl GNU Mailutils is free software; you can redistribute it and/or
dnl modify it under the terms of the GNU General Public License as
dnl published by the Free Software Foundation; either version 3, or (at
dnl your option) any later version.
dnl
dnl This program is distributed in the hope that it will be useful, but
dnl WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program.  If not, see <http://www.gnu.org/licenses/>.

dnl AM_GNU_MAILUTILS(minversion, link-req, [act-if-found], [ac-if-not-found])
dnl                      $1         $2           $3              $4
dnl Verify if GNU Mailutils is installed and if its version is `minversion'
dnl or newer.  If not installed, execute `ac-if-not-found' or, if it is not
dnl given, spit out an error message.
dnl
dnl If Mailutils is found, set:
dnl     MAILUTILS_CONFIG to the full name of the mailutils-config program;
dnl     MAILUTILS_VERSION to the Mailutils version (string);
dnl     MAILUTILS_VERSION_NUMBER to the packed numeric representation of the
dnl         GNU Mailutils version (major * 1000 + minor * 100 + patch);
dnl     MAILUTILS_LIBS to the list of cc(1) flags needed to link in the
dnl         libraries requested by `link-req';
dnl     MAILUTILS_INCLUDES to the list of cc(1) flags needed to set include
dnl         paths to the Mailutils headers.
dnl
dnl Finally, if `act-if-found' is given, execute it.  Otherwise, append the
dnl value of $MAILUTILS_LIBS to LIBS. 
dnl
AC_DEFUN([AM_GNU_MAILUTILS],
 [AC_PATH_PROG(MAILUTILS_CONFIG, mailutils-config, none, $PATH)
  if test "$MAILUTILS_CONFIG" = "none"; then
    ifelse($4,,[AC_MSG_ERROR(cannot find GNU Mailutils)], [$4])
  fi
  AC_SUBST(MAILUTILS_CONFIG)
  AC_SUBST(MAILUTILS_VERSION)
  AC_SUBST(MAILUTILS_INCLUDES)
  AC_SUBST(MAILUTILS_LIBS)
  MAILUTILS_VERSION=`$MAILUTILS_CONFIG --info version|sed 's/VERSION=//'`
  VEX=`echo $MAILUTILS_VERSION | sed 's/\./ \\\\* 1000 + /;s/\./ \\\\* 100 + /'`
  MAILUTILS_VERSION_NUMBER=`eval expr "$VEX"`
  AC_SUBST(MAILUTILS_VERSION_NUMBER)
  AC_DEFINE_UNQUOTED(MAILUTILS_VERSION, "$MAILUTILS_VERSION", [Mailutils version number]) 
  AC_DEFINE_UNQUOTED(MAILUTILS_VERSION_NUMBER, $MAILUTILS_VERSION_NUMBER,
                     [Packed Mailutils version number])
  ifelse($1,,,[
   VEX=`echo $1 | sed 's/\./ \\\\* 1000 + /;s/\./ \\\\* 100 + /'`
   min=`eval expr "$VEX"`
   if test $MAILUTILS_VERSION_NUMBER -lt $min; then
     AC_MSG_ERROR([Mailutils version too old; required is at least ]$1)
   fi])
  req=""
  for x in $2
  do
    case $x in
    mailer)   test $MAILUTILS_VERSION_NUMBER -ge 1200 && req="$req $x";;
    cfg|argp) test $MAILUTILS_VERSION_NUMBER -ge 1290 && req="$req $x";;
    *)        req="$req $x"
    esac
  done
  MAILUTILS_LIBS=`$MAILUTILS_CONFIG --link $req`
  MAILUTILS_INCLUDES=`$MAILUTILS_CONFIG --compile`
  ifelse($3,,[LIBS="$LIBS $MAILUTILS_LIBS"], [$3])
])  
  
