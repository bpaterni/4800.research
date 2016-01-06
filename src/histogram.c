#include <stdio.h>
#include <stdlib.h>

#include <glib.h>
#include <glib/gstdio.h>

#include <CL/cl.h>

#include "utils.h"

void
check(cl_int status) {
    if(status != CL_SUCCESS) {
        printf("OpenCL error (%d)\n", status);
        exit(EXIT_FAILURE);
    }
}

static GOptionEntry opt_entries[] =
{
    { NULL }
};

int
main(int argc, char *argv[]) {
    PARSE_OPTS_WITH_ENTRIES("- compute histogram of image", opt_entries)

    return EXIT_SUCCESS;
}
