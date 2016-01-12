#include <stdio.h>
#include <stdlib.h>

#include <glib.h>
#include <glib/gstdio.h>

#include <CL/cl.h>

#include "gold.h"
#include "utils.h"
#include "bmp-utils.h"

static gchar *_bmp_filename = "data/cat.bmp";

static GOptionEntry opt_entries[] =
{
    {
        "file",
        'f',
        G_OPTION_FLAG_NONE,
        G_OPTION_ARG_FILENAME,
        &_bmp_filename,
        "input BMP image used for rotation",
        "FILE"
    },
    { NULL }
};

int
main(int argc, char *argv[]) {
    PARSE_OPTS_WITH_ENTRIES("- compute histogram of image", opt_entries);

    const float theta = 45.0f;

    int rows, cols;

    float *h_input_bmp = readBmpFloat(_bmp_filename, &rows, &cols);

    const int    n_pixels = rows * cols;
    const size_t sz_image = n_pixels * sizeof(float);

    float *h_output_bmp = (float*)malloc(sz_image);
    if( !h_output_bmp ) {
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

    /* The image descriptor describes the type and dimensions of the image.
     * Here we initialize a 2D image with no pitch. */
    cl_image_desc desc = {
        .image_type           = CL_MEM_OBJECT_IMAGE2D,
        .image_width          = cols,
        .image_height         = rows,
        .image_depth          = 0,
        .image_array_size     = 0,
        .image_row_pitch      = 0,
        .image_slice_pitch    = 0,
        .num_mip_levels = 0,
        .num_samples    = 0,
        .buffer         = NULL
    };

    /* The image format describes the properties of each pixel */
    cl_image_format format = {
        .image_channel_order = CL_R,
        .image_channel_data_type  = CL_FLOAT,
    };

    /* Create input/output images */
    cl_mem img_in = clCreateImage(
            ctx,
            CL_MEM_READ_ONLY,
            &format,
            &desc,
            NULL,
            NULL);
    cl_mem img_out = clCreateImage(
            ctx,
            CL_MEM_WRITE_ONLY,
            &format,
            &desc,
            NULL,
            NULL);

    // offset to begin copy
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { cols, rows, 1 };

    clEnqueueWriteImage(
            cmdq,
            img_in,
            CL_TRUE,
            origin,
            region, 
            0, // row pitch
            0, // slice pitch
            h_input_bmp,
            0,
            NULL,
            NULL);

    gchar *source;
    if(!g_file_get_contents(
                "cl_kernels/image-rotation.cl",
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
    kernel = clCreateKernel(program, "rotation", &status);
    check(status);

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_in);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_out);
    status |= clSetKernelArg(kernel, 2, sizeof(int),    &cols);
    status |= clSetKernelArg(kernel, 3, sizeof(int),    &rows);
    status |= clSetKernelArg(kernel, 4, sizeof(float),  &theta);
    check(status);

    size_t sz_gbl_work[2] = {
        cols,
        rows
    };
    size_t sz_loc_work[2] = {
        8,
        8
    };

    status = clEnqueueNDRangeKernel(
            cmdq,
            kernel,
            2,
            NULL,
            sz_gbl_work,
            sz_loc_work,
            0,
            NULL,
            NULL);
    check(status);

    status = clEnqueueReadImage(
            cmdq,
            img_out,
            CL_TRUE,
            origin,
            region,
            0, // row-pitch
            0, // slice-pitch
            h_output_bmp,
            0,
            NULL,
            NULL);
    check(status);

    writeBmpFloat(h_output_bmp, "rotated-cat.bmp", rows, cols, _bmp_filename);

    free(source);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdq);
    clReleaseMemObject(img_in);
    clReleaseMemObject(img_out);
    clReleaseContext(ctx);

    free(h_input_bmp);
    free(h_output_bmp);

    return EXIT_SUCCESS;
}
