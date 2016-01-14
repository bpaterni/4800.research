#include "research.h"

static gchar *_bmp_filename = "data/cat.bmp";

static float gaussian_blur_filter[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
};
static const int gaussian_blur_filter_width = 5;

static float emboss_filter[9] = {
    2.0f,  0.0f,  0.0f,
    0.0f, -1.0f,  0.0f,
    0.0f,  0.0f, -1.0f
};
static const int emboss_filter_width = 3;

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
    parse_opts_with_desc_entries(
            "- compute histogram of image",
            opt_entries,
            &argc,
            &argv);

    int     filter_width = emboss_filter_width;
    float  *filter       = emboss_filter;
    size_t  sz_filter   = filter_width*filter_width*sizeof(float);

    int rows, cols;
    float *h_input_bmp = readBmpFloat(_bmp_filename, &rows, &cols);
    const int    n_pixels = rows * cols;
    const size_t sz_image = n_pixels * sizeof(float);
    //printf("bmp size: %dx%d\n", cols, rows);

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
        .num_mip_levels       = 0,
        .num_samples          = 0,
        .buffer               = NULL
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

    /* Create a buffer for the filter */
    cl_mem buf_filter = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            sz_filter,
            NULL,
            &status);
    check(status);

    status = clEnqueueWriteBuffer(
            cmdq,
            buf_filter,
            CL_TRUE,
            0,
            sz_filter,
            filter,
            0,
            NULL,
            NULL);
    check(status);

    cl_sampler sampler = clCreateSampler(
            ctx,
            CL_FALSE,
            CL_ADDRESS_CLAMP_TO_EDGE,
            CL_FILTER_NEAREST,
            &status);
    check(status);

    gchar *source;
    if(!g_file_get_contents(
                "cl_kernels/image-convolution.cl",
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
    kernel = clCreateKernel(program, "convolution", &status);
    check(status);

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem),     &img_in);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem),     &img_out);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem),     &buf_filter);
    status |= clSetKernelArg(kernel, 3, sizeof(int),        &filter_width);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_sampler), &sampler);
    check(status);

    /* This will fail if using OpenCL < 2.0 and (cols,rows) is not a multiple
     * of the local work size */
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

    writeBmpFloat(h_output_bmp, "filtered-cat.bmp", rows, cols, _bmp_filename);

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
