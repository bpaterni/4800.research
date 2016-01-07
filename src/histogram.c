#include <stdio.h>
#include <stdlib.h>

#include <glib.h>
#include <glib/gstdio.h>

#include <CL/cl.h>

#include "gold.h"
#include "utils.h"
#include "bmp-utils.h"

static const int HIST_BINS = 256;

static gchar *_bmp_filename = "data/cat.bmp";

static GOptionEntry opt_entries[] =
{
    {
        "file",
        'f',
        G_OPTION_FLAG_NONE,
        G_OPTION_ARG_FILENAME,
        &_bmp_filename,
        "BMP file for which to compute histogram",
        "FILE"
    },
    { NULL }
};

int
main(int argc, char *argv[]) {
    PARSE_OPTS_WITH_ENTRIES("- compute histogram of image", opt_entries);

    int rows, cols;

    int *h_bmp = readBmp(_bmp_filename, &rows, &cols);

    const int    n_pixels = rows * cols;
    const size_t sz_image = n_pixels * sizeof(int);

    const size_t sz_hist = HIST_BINS * sizeof(int);

    int *h_out_hist = (int*)malloc(sz_hist);
    if( !h_out_hist ) {
        exit(EXIT_FAILURE);
    }

    cl_int status;

    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    check(status);

    cl_device_id dev;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    check(status);

    cl_context ctx;
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &status);
    check(status);

    cl_command_queue cmdq;
    cmdq = clCreateCommandQueue(ctx, dev, 0, &status);
    check(status);

    cl_mem buf_image;
    cl_mem buf_out_hist;
    buf_image = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz_image, NULL, &status);
    check(status);
    buf_out_hist = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz_hist, NULL, &status);
    check(status);

    status = clEnqueueWriteBuffer(
            cmdq,
            buf_image,
            CL_TRUE,
            0,
            sz_image,
            h_bmp,
            0,
            NULL,
            NULL);
    check(status);

    int zero = 0;
    status = clEnqueueFillBuffer(
            cmdq,
            buf_out_hist,
            &zero,
            sizeof(int),
            0,
            sz_hist,
            0,
            NULL,
            NULL);
    check(status);

    gchar *source;
    if(!g_file_get_contents(
                "cl_kernels/histogram.cl",
                &source,
                NULL,
                NULL)) {
        printf("Unable to read CL source file\n");
        exit(EXIT_FAILURE);
    }

    cl_program program = clCreateProgramWithSource(
            ctx,
            1,
            (const char **)&source,
            NULL,
            &status);
    check(status);

    status = clBuildProgram(program, 1, &dev, NULL, NULL, NULL);
    if(status != CL_SUCCESS) {
        print_cl_compiler_error(program, dev);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel;
    kernel = clCreateKernel(program, "histogram", &status);
    check(status);

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_image);
    status |= clSetKernelArg(kernel, 1, sizeof(int), &sz_image);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out_hist);
    check(status);

    size_t sz_gbl_work[1];
    sz_gbl_work[0] = 1024;

    size_t sz_loc_work[1];
    sz_loc_work[0] = 64;

    status = clEnqueueNDRangeKernel(
            cmdq,
            kernel,
            1,
            NULL,
            sz_gbl_work,
            sz_loc_work,
            0,
            NULL,
            NULL);
    check(status);

    status = clEnqueueReadBuffer(
            cmdq,
            buf_out_hist,
            CL_TRUE,
            0,
            sz_hist,
            h_out_hist,
            0,
            NULL,
            NULL);
    check(status);

    int *ref_histogram;
    ref_histogram  = histogramGold(h_bmp, rows*cols, HIST_BINS);
    int i;
    for(i=0; i<HIST_BINS; i++) {
        printf("%3d histogram(reference): %10d : %10d\n",
                i,
                h_out_hist[i],
                ref_histogram[i]);
    }
    free(ref_histogram);

    free(source);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdq);
    clReleaseMemObject(buf_image);
    clReleaseMemObject(buf_out_hist);
    clReleaseContext(ctx);

    free(h_bmp);
    free(h_out_hist);

    return EXIT_SUCCESS;
}
