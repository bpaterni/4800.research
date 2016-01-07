#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

void
check(cl_int status) {
    if(status != CL_SUCCESS) {
        printf("OpenCL error (%d)\n", status);
        exit(EXIT_FAILURE);
    }
}

void
print_cl_compiler_error(cl_program program, cl_device_id dev) {
    cl_int status;

    size_t sz_log;
    status = clGetProgramBuildInfo(
            program,
            dev,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &sz_log);
    check(status);

    char *log = (char*)malloc(sz_log);
    if(!log) {
        exit(EXIT_FAILURE);
    }

    status = clGetProgramBuildInfo(
            program,
            dev,
            CL_PROGRAM_BUILD_LOG,
            sz_log,
            log,
            NULL);
    check(status);

    printf("%s\n", log);
    free(log);
}
