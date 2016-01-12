AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CFLAGS   = $(GLIB_CFLAGS)
AM_LDFLAGS  = $(GLIB_LIBS) \
	      $(OPENCL_LIB)

bin_PROGRAMS = vecadd \
	       histogram \
	       rotimage

vecadd_SOURCES = vecadd.c

histogram_SOURCES = histogram.c \
		    $(top_srcdir)/utils/gold.c \
		    $(top_srcdir)/utils/bmp-utils.c \
		    $(top_srcdir)/utils/utils.c

rotimage_SOURCES = image-rotation.c \
		   $(top_srcdir)/utils/bmp-utils.c \
		   $(top_srcdir)/utils/utils.c
