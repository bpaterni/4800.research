#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([csbs-research], [0.1], [bpaterni@uni.edu])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/vecadd.c])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC

# Checks for libraries.
AC_CHECK_LIB([OpenCL],
	     [clGetPlatformIDs],
	     [OPENCL_LIB=-lOpenCL],
	     [AC_MSG_ERROR([Could not find OpenCL library])])
AC_SUBST([OPENCL_LIB])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC

PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.42.1])

AC_CONFIG_FILES([
  Makefile
  src/Makefile
  ])
AC_OUTPUT
