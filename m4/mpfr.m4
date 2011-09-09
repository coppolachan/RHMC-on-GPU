AC_DEFUN([MPFR_DEVEL],
[
AC_CACHE_CHECK([for MPFR devel stuff], ac_cv_mpfr_devel,
[
	#
	# Set up configure script macros
	#

AC_ARG_WITH([mpfr],
AC_HELP_STRING([--with-mpfr=PATH],
   		[Path containing the MPFR Multiple Precision library [default=/usr/lib or /usr/local/lib]]),
		[MPFR_lib_check="$with_mpfr/lib64 $with_mpfr/lib"
		 MPFR_inc_check="$with_mpfr/include"],
		[MPFR_lib_check="/usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib"
		 MPFR_inc_check="/usr/local/include /usr/include"]
   		)
		
	#
	# Look for MPFR library
	#
	AC_MSG_CHECKING([for MPFR library directory])
	MPFR_libdir=
	for dir in $MPFR_lib_check
	do
		if test -d "$dir" && \
			( test -f "$dir/libmpfr.so" ||
			  test -f "$dir/libmpfr.a" )
		then
			MPFR_libdir=$dir
			break
		fi
	done

	if test -z "$MPFR_libdir"
	then
		AC_MSG_ERROR([[Didn't find the MPFR library dir in '$MPFR_lib_check']
	                      [Check path if already installed"]
	                      [Download at: http://www.mpfr.org/"]])
	fi

	case "$MPFR_libdir" in
		/* ) ;;
		* )  AC_MSG_ERROR([The MPFR library directory ($MPFR_libdir) must be an absolute path.]) ;;
	esac

	AC_MSG_RESULT([$MPFR_libdir])

	case "$MPFR_libdir" in
	  /usr/lib) ;;
	  *) MPFR_LIB="-L${MPFR_libdir} -lmpfr -lgmp" ;;
	esac


	#
	# Look for MPFR headers
	#
	AC_MSG_CHECKING([for MPFR header directory])
	MPFR_incdir=
	for dir in $MPFR_inc_check
	do
		if test -d "$dir" && test -f "$dir/mpfr.h"
		then
			MPFR_incdir=$dir
			break
		fi
	done

	if test -z "$MPFR_incdir"
	then
		AC_MSG_ERROR([Didn't find the MPFR header dir in '$MPFR_inc_check'])
	fi

	case "$MPFR_incdir" in
		/* ) ;;
		* )  AC_MSG_ERROR([The MPFR header directory ($MPFR_incdir) must be an absolute path.]) ;;
	esac

	AC_MSG_RESULT([$MPFR_incdir])

	MPFR_INCFLAGS="-I${MPFR_incdir}"


	AC_MSG_CHECKING([that we can build MPFR programs])
	AC_COMPILE_IFELSE(
		[AC_LANG_PROGRAM(
		[[#ifdef __cplusplus]
		 [extern "C"]
		 [#endif]
		 [#include <mpfr.h>]],
		[mpfr_t x;
		[mpfr_init(x);]])],
		ac_cv_mpfr_devel=yes,
		AC_MSG_ERROR(no))

])])