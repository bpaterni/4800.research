AM_CPPFLAGS = -I$(top_srcdir)/include

bin_PROGRAMS = vecadd \
	       histogram

vecadd_CFLAGS  = $(GLIB_CFLAGS)
vecadd_SOURCES = vecadd.c
vecadd_LDADD   = $(GLIB_LIBS) $(OPENCL_LIB)

histogram_CFLAGS = $(GLIB_CFLAGS)
histogram_SOURCES = histogram.c \
		    $(top_srcdir)/utils/utils.c
histogram_LDADD   = $(GLIB_LIBS) $(OPENCL_LIB)
