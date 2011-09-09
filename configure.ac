#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT(rhmc_staggered_cuda, 0.1.3, cossu@post.kek.jp)
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([1.9 -Wall -Werror foreign])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile lib/Makefile])

m4_include([m4/mpfr.m4])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
	
#---------------------------------------------------------
# Checks for CUDA
#---------------------------------------------------------
AC_ARG_ENABLE([cuda], 
	      AC_HELP_STRING([--enable-cuda=ARCH],
	      [Enable Nvidia CUDA based modules for architecture ARCH (see nvcc option -arch and README file).]),[

	      using_cuda=$enableval

	      NVCCFLAGS=""
	      # Search for compiler
	      AC_CHECK_PROG([cuda_compiler], [nvcc], yes, no) 
	      AS_IF([test "x$cuda_compiler" = "xno"],[
                        AC_MSG_ERROR([NVCC compiler not found! Aborting compilation.])
              ])
	      AC_PATH_PROG(NVCC,nvcc, "no")
	      CUDA_COMPILE="-DUSE_GPU"		     
	      AC_SUBST(CUDA_COMPILE)

	      # Check nvcc version, should be >=3.x
              AS_IF([(nvcc --version | grep -q "release 3.") || (nvcc --version | grep -q "release 4.")],
              		[],
                        [AC_MSG_WARN([NVCC compiler version is NOT >=3.x!])
              ])
	      
	      # If $enable_cuda is not empty, set to CUDA_ARCH to
              # supplied value, else set to value sm_11

	      AS_IF([test "x$enableval" = "xyes"],[
                        CUDA_ARCH=" -arch=sm_11"
			echo "compiling with nvcc for architecture sm_11 (default)"
           		],[
                	CUDA_ARCH=" -arch=$enableval"
			echo "compiling with nvcc for architecture $enableval"
              ])

# --------------------------------------------------------
# Setup CUDA paths
# --------------------------------------------------------
		AC_ARG_WITH([cuda],
   		AC_HELP_STRING([--with-cuda=PATH],
   		[Prefix where CUDA is installed [default=/usr/local/cuda]])
   		)

		if test -n "$with_cuda"
		   then
   		   CUDA_CFLAGS="-I$with_cuda/include"
   		   CUDA_LIBS="-L$with_cuda/lib"

   		   # If $build_cpu contains "_64", append "64" to CUDA_LIBS
   		   AS_IF([echo $build_cpu | grep -q "_64"],
         	   	       [CUDA_LIBS+="64"])
		   # Append " -lcudart" to CUDA_LIBS
   		   CUDA_LIBS+=" -lcudart"
   		   NVCC="$with_cuda/bin/nvcc"
		else	
		   CUDA_CFLAGS="-I/usr/local/cuda/include"
   		   CUDA_LIBS="-L/usr/local/cuda/lib"

   		   # If $build_cpu contains "_64", append "64" to CUDA_LIBS
     		   AS_IF([echo $build_cpu | grep -q "_64"],
         	   	       [CUDA_LIBS+="64"])
		       # Append " -lcudart" to CUDA_LIBS
		   CUDA_LIBS+=" -lcudart"
   		   NVCC="nvcc"
		 fi

		AC_ARG_WITH([cuda_comp],
   		AC_HELP_STRING([--with-cuda_comp=PATH],
		[GCC compiler PATH to be used. Use in case your default compiler is incompatible with nvcc [default= gcc]])
)
		
		if test -n "$with_cuda_comp"
		   then
   		   CUDA_COMPILER=" -ccbin $with_cuda_comp"
		fi		


AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
])
AM_CONDITIONAL([ENABLE_CUDA], [test -n "$using_cuda"])
# put here a test for cuda compilation
# -----------------------------------------------------
# Setup nvcc flags
# -----------------------------------------------------
# Check whether to use device emulation mode for cuda (if no cuda capable gpu is available)
AC_ARG_ENABLE([emu],
        AS_HELP_STRING([--enable-emu], [Enable device emulation for cuda modules (nvcc version <= 3.0 only).]),
        [EMULATION=true],
        [EMULATION=false])

# If debug flag is set apply debugging compilation flags, otherwise build compilation flags
AS_IF([test "x$DEBUG" = "xtrue"],
      [NVCCFLAGS="-g --compiler-options -fno-strict-aliasing --compiler-options -fno-inline"],
      [NVCCFLAGS="-O3 -m64 --compiler-options -fno-strict-aliasing"])

 # Add architecture to flags
 NVCCFLAGS+=" $CUDA_ARCH $CUDA_COMPILER"

 # If device emulation was set, add deviceemu flag
 AS_IF([test "x$EMULATION" = "xtrue"],
       [NVCCFLAGS+=" -deviceemu"])

# Make NVCCFLAGS available in Makefile.am
AC_SUBST(NVCCFLAGS)



#####################################
MPFR_DEVEL()
AC_SUBST(MPFR_LIB)
AC_SUBST(MPFR_INCFLAGS)


# Checks for header files.
AC_CHECK_HEADERS([inttypes.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_SIZE_T
#AC_TYPE_UINT32_T
#AC_TYPE_UINT64_T

# Checks for library functions.
AC_CHECK_FUNCS([memset pow sqrt])

AC_OUTPUT
