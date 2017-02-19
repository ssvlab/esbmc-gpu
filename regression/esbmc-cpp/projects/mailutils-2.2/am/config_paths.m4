#serial 1
dnl Copyright (C) 1996, 1997, 1998, 2002, 2004, 2005, 2007, 2009, 2010
dnl Free Software Foundation, Inc.
dnl
dnl Written by Miles Bader <miles@gnu.ai.mit.edu> and
dnl Sergey Poznyakoff <gray@gnu.org>
dnl
dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 3, or (at your option)
dnl any later version.
dnl
dnl This program is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl GNU General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program.  If not, see <http://www.gnu.org/licenses/>.
dnl

dnl MU_CONFIG_PATHS -- Configure system paths for use by programs
dnl   $1 - PATHS    -- The file to read containing the paths
dnl   $2 - HDRDEFS  -- The file to put the generated C defines into
dnl   $3 - MAKEDEFS -- [OPTIONAL] The file to put the generated 
dnl                    make `PATHDEF_' vars
dnl
dnl From the paths listed in the file PATHS, generate a C header file
dnl HDRDEFS, containing the necessary #define statements.
dnl
dnl Additionally, if MAKEDEFS is supplied, populate this file with
dnl make variables.  A variable for each PATH_FOO will be called
dnl PATHDEF_FOO and will be set to a cpp option to define that path,
dnl unless it is to be defined using a system define.
dnl
dnl Use of MAKEDEFS is recommended if any of the settings from PATHS refer
dnl to environment or make variables.
dnl

AC_DEFUN([MU_CONFIG_PATHS], [
  dnl We need to know if we're cross compiling.
  AC_REQUIRE([AC_PROG_CC])

  AC_CHECK_HEADER(paths.h, AC_DEFINE(HAVE_PATHS_H, 1,
        [Define if you have the <paths.h> header file]) mu_paths_h="<paths.h>")

  dnl A slightly bogus use of AC_ARG_WITH; we never actually use
  dnl $with_PATHVAR, we just want to get this entry put into the help list.
  dnl We actually look for `with_' variables corresponding to each path
  dnl configured.
  AC_ARG_WITH([PATHVAR],
              AC_HELP_STRING([--with-PATHVAR=PATH],
                             [Set the value of PATHVAR to PATH
                          PATHVAR is the name of a \`PATH_FOO' variable,
                          downcased, with \`_' changed to \`-']))

  # For case-conversion with sed
  MU_UCASE=ABCDEFGHIJKLMNOPQRSTUVWXYZ
  mu_lcase=abcdefghijklmnopqrstuvwxyz

  tmpdir="$TMPDIR"
  test x"$tmpdir" = x && tmpdir="/tmp"
  mu_cache_file="$tmpdir/,mu-path-cache.$$"
  mu_tmp_file="$tmpdir/,mu-tmp.$$"
  ac_clean_files="$ac_clean_files $mu_cache_file $mu_tmp_file"
  while read mu_path mu_search; do
    test "$mu_path" = "#" -o -z "$mu_path" && continue

    mu_pathvar="`echo $mu_path  | sed y/${MU_UCASE}/${mu_lcase}/`"
    AC_MSG_CHECKING(for value of $mu_path)

    mu_val='' mu_hdr='' mu_sym=''
    mu_cached='' mu_defaulted=''
    mu_cross_conflict=''
    if test "`eval echo '$'{with_$mu_pathvar+set}`" = set; then
      # User-supplied value
      eval mu_val=\"'$'with_$mu_pathvar\"
    elif test "`eval echo '$'{mailutils_cv_$mu_pathvar+set}`" = set; then
      # Cached value
      eval mu_val=\"'$'mailutils_cv_$mu_pathvar\"
      # invert escaped $(...) notation used in autoconf cache
      eval mu_val=\"\`echo \'"$mu_val"\' \| sed \''s/@(/$\(/g'\'\`\"
      mu_cached="(cached) "
    elif test "`eval echo '$'{mailutils_cv_hdr_$mu_pathvar+set}`" = set; then
      # Cached non-value
      eval mu_hdr=\"'$'mailutils_cv_hdr_$mu_pathvar\"
      eval mu_sym=\"'$'mailutils_cv_hdr_sym_$mu_pathvar\"
      mu_cached="(cached) "
    else
      # search for a reasonable value

      mu_test_type=r		# `exists'
      mu_default='' mu_prev_cross_test=''
      for mu_try in $mu_paths_h $mu_search; do
	mu_cross_test=''
	case "$mu_try" in
	  "<"*">"*)
	    # <HEADER.h> and <HEADER.h>:SYMBOL -- look for SYMBOL in <HEADER.h>
	    # SYMBOL defaults to _$mu_path (e.g., _PATH_FOO)
	    changequote(,)	dnl Avoid problems with [ ] in regexps
	    eval mu_hdr=\'`echo "$mu_try" |sed 's/:.*$//'`\'
	    eval mu_sym=\'`echo "$mu_try" |sed -n 's/^<[^>]*>:\(.*\)$/\1/p'`\'
	    changequote([,])
	    test "$mu_sym" || mu_sym="_$mu_path"
	    AC_EGREP_CPP(HAVE_$mu_sym,
[#include ]$mu_hdr[
#ifdef $mu_sym
HAVE_$mu_sym
#endif],
	      :, mu_hdr='' mu_sym='')
	    ;;

	  search:*)
	    # Do a path search.  The syntax here is: search:NAME[:PATH]...

	    # Path searches always generate potential conflicts
	    test "$cross_compiling" = yes && { mu_cross_conflict=yes; continue; }

	    changequote(,)	dnl Avoid problems with [ ] in regexps
	    mu_name="`echo $mu_try | sed 's/^search:\([^:]*\).*$/\1/'`"
	    mu_spath="`echo $mu_try | sed 's/^search:\([^:]*\)//'`"
	    changequote([,])

	    test "$mu_spath" || mu_spath="$PATH"

	    for mu_dir in `echo "$mu_spath" | sed 'y/:/ /'`; do
	      test -z "$mu_dir" && mu_dir=.
	      if test -$mu_test_type "$mu_dir/$mu_name"; then
		mu_val="$mu_dir/$mu_name"
		break
	      fi
	    done
	    ;;

	  no) mu_default=no;;
	  x|d|f|c|b) mu_test_type=$mu_try;;

	  *)
	    # Just try the given name, with make-var substitution.  Besides 
	    # yielding a value if found, this also sets the default.

	    case "$mu_try" in "\""*"\"")
	      # strip off quotes
	      mu_try="`echo $mu_try | sed -e 's/^.//' -e 's/.$//'`"
	    esac

	    test -z "$mu_default" && mu_default="$mu_try"
	    test "$cross_compiling" = yes && { mu_cross_test=yes; continue; }

	    # See if the value begins with a $(FOO)/${FOO} make variable
	    # corresponding to a shell variable, and if so set try_exp to the
	    # value thereof.  Recurse.
	    mu_try_exp="$mu_try"
	    changequote(,)
	    mu_try_var="`echo "$mu_try_exp" |sed -n 's;^\$[({]\([-_a-zA-Z]*\)[)}].*;\1;p'`"
	    while eval test \"$mu_try_var\" && eval test '${'$mu_try_var'+set}'; do
	      # yes, and there's a corresponding shell variable, which substitute
	      if eval test \"'$'"$mu_try_var"\" = NONE; then
		# Not filled in by configure yet
		case "$mu_try_var" in
		  prefix | exec_prefix)
		    mu_try_exp="$ac_default_prefix`echo "$mu_try_exp" |sed 's;^\$[({][-_a-zA-Z]*[)}];;'`";;
		esac
		mu_try_var=''	# Stop expansion here
	      else
		# Use the actual value of the shell variable
		eval mu_try_exp=\"`echo "$mu_try_exp" |sed 's;^\$[({]\([-_a-zA-Z]*\)[)}];\$\1;'`\"
		mu_try_var="`echo "$mu_try_exp" |sed -n 's;^\$[({]\([-_a-zA-Z]*\)[)}].*;\1;p'`"
	      fi
	    done
	    changequote([,])

	    test -$mu_test_type "$mu_try_exp" && mu_val="$mu_try"
	    ;;

	esac

	test "$mu_val" -o "$mu_hdr" && break
	test "$mu_cross_test" -a "$mu_prev_cross_test" && mu_cross_conflict=yes
	mu_prev_cross_test=$mu_cross_test
      done

      if test -z "$mu_val" -a -z "$mu_hdr"; then
	if test -z "$mu_default"; then
	  mu_val=no
	else
	  mu_val="$mu_default"
	  mu_defaulted="(default) "
	fi
      fi
    fi

    if test "$mu_val"; then
      AC_MSG_RESULT(${mu_cached}${mu_defaulted}$mu_val)
      test "$mu_cross_conflict" -a "$mu_defaulted" \
	&& AC_MSG_WARN(may be incorrect because of cross-compilation)
      # Put the value in the autoconf cache.  We replace $( with @( to avoid
      # variable evaluation problems when autoconf reads the cache later.
      echo mailutils_cv_$mu_pathvar=\'"`echo "$mu_val" | sed 's/\$(/@(/g'`"\'
    elif test "$mu_hdr"; then
      AC_MSG_RESULT(${mu_cached}from $mu_sym in $mu_hdr)
      echo mailutils_cv_hdr_$mu_pathvar=\'"$mu_hdr"\'
      echo mailutils_cv_hdr_sym_$mu_pathvar=\'"$mu_sym"\'
    fi
  done <[$1] >$mu_cache_file

  # Read the cache values constructed by the previous loop, 
  . $mu_cache_file

  # Generate a file of #ifdefs that defaults PATH_FOO macros to _PATH_FOO (or
  # some other symbol) (excluding any who's value is set to `no').
  while read mu_cache_set; do
    mu_sym_var="`echo "$mu_cache_set" | sed 's/=.*$//'`"
    eval mu_sym_value=\"'$'"$mu_sym_var"\"
    case $mu_sym_var in
    mailutils_cv_hdr_sym_*)
      mu_path="`echo $mu_sym_var | sed -e 's/^mailutils_cv_hdr_sym_//' -e y/${mu_lcase}/${MU_UCASE}/`"
      ;;
    mailutils_cv_path_*)
      mu_path="PATH_`echo $mu_sym_var | sed -e 's/^mailutils_cv_path_//' -e y/${mu_lcase}/${MU_UCASE}/`"
      s=`echo "$mu_sym_value" | sed 's/[[^@\$]]//g'`
      if test -n "$s"; then
        continue
      fi
      mu_sym_value="\"$mu_sym_value\""
      ;;
    *)
      continue;;
    esac
    cat <<EOF
#ifndef $mu_path
#define $mu_path $mu_sym_value
#endif
EOF
  done < $mu_cache_file >$[$2]
  AC_SUBST_FILE([$2])

  m4_if([$3],[],[],[
  # Construct the pathdefs file -- a file of make variable definitions, of
  # the form PATHDEF_FOO, that contain cc -D switches to define the cpp macro
  # PATH_FOO.
  grep -v '^mailutils_cv_hdr_' < $mu_cache_file | \
  while read mu_cache_set; do
    mu_var="`echo $mu_cache_set | sed 's/=.*$//'`"
    eval mu_val=\"'$'"$mu_var"\"
    # invert escaped $(...) notation used in autoconf cache
    eval mu_val=\"\`echo \'"$mu_val"\' \| sed \''s/@(/$\(/g'\'\`\"
    if test "$mu_val" != no; then
      mu_path="`echo $mu_var | sed -e 's/^mailutils_cv_//' -e y/${mu_lcase}/${MU_UCASE}/`"
      mu_pathdef="`echo $mu_path | sed 's/^PATH_/PATHDEF_/'`"
      echo $mu_pathdef = -D$mu_path='\"'"$mu_val"'\"'
    fi
  done >$[$3]
  AC_SUBST_FILE([$3])
  ]) ])
