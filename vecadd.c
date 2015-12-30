// This program implements a vector addition using OpenCL
#include <stdio.h>
#include <stdlib.h>

#include <glib.h>

#include <CL/cl.h>

// OpenCL kernel to perform an element-wise addition
const char *program_source =
"__kernel\n"
"void vecadd(__global int *A,\n"
"            __global int *B,\n"
"            __global int *C)\n"
"{\n"
"  // Get the work-item's unique ID\n"
"  int idx = get_global_id(0);\n"
"\n"
"  // Add the corresponding locations of\n"
"  // 'A' and 'B', and store the result in 'C'.\n"
"  C[idx] = A[idx] + B[idx];\n"
"}\n";

long elements = 2048;

static GOptionEntry entries[] =
{
    {
        "elements", 'l', 0, G_OPTION_ARG_INT64, &elements,
        "add together vectors of size N", "N"
    },
    { NULL }
};

int
main(int argc, char *argv[]) {
    GError *error = NULL;
    GOptionContext *ctx_opts;

    ctx_opts = g_option_context_new("- test vector addition performance");
    g_option_context_add_main_entries(ctx_opts, entries, NULL);
    if( !g_option_context_parse(ctx_opts, &argc, &argv, &error) ) {
        g_print("option parsing failed: %s\n", error->message);
        exit(EXIT_FAILURE);
    }

    size_t datasize = sizeof(int)*elements;

    int *A = (int*)malloc(datasize); // Input array
    int *B = (int*)malloc(datasize); // Input array
    int *C = (int*)malloc(datasize); // Output array

    int i;
    for(i=0; i<elements; i++) {
        A[i] = i;
        B[i] = i;
    }

    cl_int status;

    const size_t NUM_PLATFORMS = 1;
    cl_platform_id platform;
    status = clGetPlatformIDs(NUM_PLATFORMS, &platform, NULL);

    const cl_uint NUM_DEVICES = 4;
    cl_device_id  devices[NUM_DEVICES];
    cl_uint       num_devices;
    status = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            NUM_DEVICES,
            devices,
            &num_devices);

    cl_context ctx = clCreateContext(
            NULL,
            num_devices,
            devices,
            NULL, NULL,
            &status);

    cl_uint idx;
    for(idx=0; idx<num_devices; idx++) {
        cl_device_id dev = devices[idx];

        char dev_name[256];

        size_t len_dev_name;
        status = clGetDeviceInfo(
                dev,
                CL_DEVICE_NAME,
                256,
                dev_name,
                &len_dev_name);

        cl_device_type dev_type;
        size_t len_dev_type;
        status = clGetDeviceInfo(
                dev,
                CL_DEVICE_TYPE,
                8,
                &dev_type,
                &len_dev_type);

        printf("Device Name %s\n", dev_name);
        printf("Device Type: %d\n", dev_type);
        printf("---\n");
    }

#if 0
    char   platform_ver[256];
    size_t len_platform_ver;
    status = clGetPlatformInfo(
            platform,
            CL_PLATFORM_VERSION,
            32,
            &platform_ver,
            &len_platform_ver);

    printf("Platform version: %s\n", platform_ver);
#endif

    clReleaseContext(ctx);

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
