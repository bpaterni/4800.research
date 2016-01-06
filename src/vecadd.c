// This program implements a vector addition using OpenCL
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

int elements = 2048;

static GOptionEntry entries[] =
{
    {
        "elements", 'l', 0, G_OPTION_ARG_INT, &elements,
        "add together vectors of size N", "N"
    },
    { NULL }
};

int
main(int argc, char *argv[]) {
    PARSE_OPTS_WITH_ENTRIES(
            "- test vector addition performance",
            entries);

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
    check(status);

    const cl_uint NUM_DEVICES = 4;
    cl_device_id  devices[NUM_DEVICES];
    cl_uint       num_devices;
    status = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            NUM_DEVICES,
            devices,
            &num_devices);
    check(status);

    cl_context ctx = clCreateContext(
            NULL,
            num_devices,
            devices,
            NULL, NULL,
            &status);
    check(status);

    cl_command_queue *cmd_qs =
        (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_devices);

    cl_uint idx;
    for(idx=0; idx<num_devices; idx++) {
        cl_device_id dev = devices[idx];

        cmd_qs[idx] = clCreateCommandQueue(
                ctx,
                dev,
                0,
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
            cmd_qs[0],
            bufB,
            CL_FALSE,
            0, datasize,
            B,
            0, NULL, NULL);

    gchar *program_source;
    if(!g_file_get_contents(
                "cl_kernels/vecadd.cl",
                &program_source,
                NULL, NULL)) {
        printf("Unable to read CL source file\n");
        exit(EXIT_FAILURE);
    }

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

    for(i=0; i<elements; i++) {
        printf("%d, ", C[i]);
    }
    printf("\n");

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    g_free(program_source);

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
