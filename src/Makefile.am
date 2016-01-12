AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CFLAGS   = $(GLIB_CFLAGS)

bin_PROGRAMS = vecadd \
	       histogram \
	       rotimage

vecadd_SOURCES = vecadd.c
vecadd_LDADD   = $(GLIB_LIBS) $(OPENCL_LIB)

histogram_SOURCES = histogram.c \
		    $(top_srcdir)/utils/gold.c \
		    $(top_srcdir)/utils/bmp-utils.c \
		    $(top_srcdir)/utils/utils.c
histogram_LDADD   = $(GLIB_LIBS) $(OPENCL_LIB)

rotimage_LDADD   = $(GLIB_LIBS) $(OPENCL_LIB)
rotimage_SOURCES = image-rotation.c \
		   $(top_srcdir)/utils/bmp-utils.c \
		   $(top_srcdir)/utils/utils.c
