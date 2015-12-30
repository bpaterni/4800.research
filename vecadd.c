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

    cl_command_queue *cmd_qs =
        (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_devices);

    cl_uint idx;
    for(idx=0; idx<num_devices; idx++) {
        cl_device_id dev = devices[idx];

        cmd_qs[idx] = clCreateCommandQueueWithProperties(
                ctx,
                dev,
                NULL,
                &status);
    }

    /* Allocate two input buffers and one output buffer for the three vectors
     * in the vector addition. */
    cl_mem bufA = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            datasize,
            NULL,
            &status);
    cl_mem bufB = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            datasize,
            NULL,
            &status);
    cl_mem bufC = clCreateBuffer(
            ctx,
            CL_MEM_WRITE_ONLY,
            datasize,
            NULL,
            &status);

    /* Write data from the input arrays to the buffers. */
    status = clEnqueueWriteBuffer(
            cmd_qs[0],
            bufA,
            CL_FALSE,
            0, datasize,
            A,
            0, NULL, NULL);
    status = clEnqueueWriteBuffer(
            cmd_qs[1],
            bufB,
            CL_FALSE,
            0, datasize,
            B,
            0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(
            ctx,
            1,
            (const char **)&program_source,
            NULL,
            &status);

    status = clBuildProgram(
            program,
            num_devices,
            devices,
            NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(
            program,
            "vecadd",
            &status);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    size_t index_space_size[1],
           work_group_size[1];

    index_space_size[0] = elements;
    work_group_size[0]  = 256;

    status = clEnqueueNDRangeKernel(
            cmd_qs[0],
            kernel,
            1,
            NULL,
            index_space_size,
            work_group_size,
            0, NULL, NULL);

    status = clEnqueueReadBuffer(
            cmd_qs[0],
            bufC,
            CL_TRUE,
            0,
            datasize,
            C,
            0, NULL, NULL);
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

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    for(idx=0; idx<num_devices; idx++) {
        clReleaseCommandQueue(cmd_qs[idx]);
    }
    free(cmd_qs);

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    clReleaseContext(ctx);

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
