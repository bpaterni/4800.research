#include "research.h"

static gchar *_bmp_filename = "data/cat.bmp";

static float gaussian_blur_filter[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
};
static float *   filter = gaussian_blur_filter;
static const int filter_width = 5;
static const int filter_sz = 25 * sizeof(float);

static float emboss_filter[9] = {
    2.0f,  0.0f,  0.0f,
    0.0f, -1.0f,  0.0f,
    0.0f,  0.0f, -1.0f
};
static const int emboss_filter_width = 3;

static const int HIST_BINS = 256;

static GOptionEntry opt_entries[] =
{
    {
        "file",
        'f',
        G_OPTION_FLAG_NONE,
        G_OPTION_ARG_FILENAME,
        &_bmp_filename,
        "input BMP image used for convolution/histogram",
        "FILE"
    },
    { NULL }
};

int
main(int argc, char *argv[]) {
    parse_opts_with_desc_entries(
            "- compute convolution/histogram of image",
            opt_entries,
            &argc,
            &argv);

    /* */
    int rows, cols;
    float *h_input_bmp = readBmpFloat(_bmp_filename, &rows, &cols);

    const int num_pixels  = rows * cols;
    const size_t image_sz = num_pixels * sizeof(float);
    //printf("bmp size: %dx%d\n", cols, rows);

    const int histogram_sz = HIST_BINS * sizeof(int);
    int *h_output_histogram = (int*)malloc(histogram_sz);
    if( !h_output_histogram ) {
        exit(EXIT_FAILURE);
    }

    cl_int status;

    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    clcheck(status);

    int has_cpu_dev = 0;
    cl_uint num_devs;
    cl_device_id devs[2];
    cl_device_id dev_gpu;
    cl_device_id dev_cpu;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev_gpu, NULL);
    clcheck(status);
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev_cpu, NULL);
    if(status == CL_DEVICE_NOT_FOUND) {
        has_cpu_dev = 0;
        num_devs = 1;
        devs[0] = dev_gpu;
    } else {
        has_cpu_dev = 1;
        clcheck(status);

        num_devs = 2;
        devs[0] = dev_gpu;
        devs[1] = dev_cpu;
    }

    cl_context ctx;
    ctx = clCreateContext(NULL, num_devs, devs, NULL, NULL, &status);
    clcheck(status);

    cl_command_queue q_gpu;
    cl_command_queue q_cpu;
    q_gpu = clCreateCommandQueue(ctx, dev_gpu, 0, &status);
    clcheck(status);
    if(has_cpu_dev) {
        q_cpu = clCreateCommandQueue(ctx, dev_cpu, 0, &status);
        clcheck(status);
    }

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
    cl_mem img_in;
    img_in = clCreateImage(
            ctx,
            CL_MEM_READ_ONLY,
            &format,
            &desc,
            NULL,
            NULL);

    /* Create buffer object for the output histogram */
    cl_mem buf_out_histogram;
    buf_out_histogram = clCreateBuffer(
            ctx,
            CL_MEM_WRITE_ONLY,
            histogram_sz,
            NULL,
            &status);
    clcheck(status);

    cl_mem buf_filter;
    buf_filter = clCreateBuffer(
            ctx,
            CL_MEM_READ_ONLY,
            filter_sz,
            NULL,
            &status);
    clcheck(status);

    cl_mem pipe;
#ifdef OCL_PIPES
    pipe = clCreatePipe(
            ctx,
            0,
            sizeof(float),
            num_pixels,
            NULL,
            &status);
    clcheck(status);
#else
    pipe = clCreateBuffer(
            ctx,
            CL_MEM_READ_WRITE,
            image_sz,
            NULL,
            &status);
    clcheck(status);
#endif  /* OCL_PIPES */

    /* origin - offset within the image to begin copy from
     * region - elements to copy per dimension */
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { cols, rows, 1 };

    /* Copy host image data to the GPU */
    clEnqueueWriteImage(
            q_gpu,
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

    /* Copy filter to GPU */
    status = clEnqueueWriteBuffer(
            q_gpu,
            buf_filter,
            CL_TRUE,
            0,
            filter_sz,
            filter,
            0,
            NULL,
            NULL);
    clcheck(status);

    int zero = 0;
    status = clEnqueueFillBuffer(
            has_cpu_dev ? q_cpu : q_gpu,
            buf_out_histogram,
            &zero,
            sizeof(int),
            0,
            histogram_sz,
            0,
            NULL,
            NULL);
    clcheck(status);

    gchar *source;
    if(!g_file_get_contents(
                "cl_kernels/producer-consumer.cl",
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
    clcheck(status);

    status = clBuildProgram(program, num_devs, devs, NULL, NULL, NULL);
    if(status != CL_SUCCESS) {
        print_cl_compiler_error(program, dev_gpu);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel_producer;
    cl_kernel kernel_consumer;
    kernel_producer = clCreateKernel(program, "producer", &status);
    clcheck(status);
    kernel_consumer = clCreateKernel(program, "consumer", &status);
    clcheck(status);

#ifdef OCL_PIPES
    status  = clSetKernelArg(kernel_producer, 0, sizeof(cl_mem),     &img_in);
    status |= clSetKernelArg(kernel_producer, 1, sizeof(cl_mem),     &pipe);
    status |= clSetKernelArg(kernel_producer, 2, sizeof(cl_mem),     &buf_filter);
    status |= clSetKernelArg(kernel_producer, 3, sizeof(int),        &filter_width);
    clcheck(status);
#else
    status  = clSetKernelArg(kernel_producer, 0, sizeof(cl_mem), &img_in);
    status |= clSetKernelArg(kernel_producer, 1, sizeof(int),    &rows);
    status |= clSetKernelArg(kernel_producer, 2, sizeof(int),    &cols);
    status |= clSetKernelArg(kernel_producer, 3, sizeof(cl_mem), &pipe);
    status |= clSetKernelArg(kernel_producer, 4, sizeof(cl_mem), &buf_filter);
    status |= clSetKernelArg(kernel_producer, 5, sizeof(int),    &filter_width);
    clcheck(status);
#endif

    status  = clSetKernelArg(kernel_consumer, 0, sizeof(cl_mem), &pipe);
    status |= clSetKernelArg(kernel_consumer, 1, sizeof(int),    &num_pixels);
    status |= clSetKernelArg(kernel_consumer, 2, sizeof(cl_mem), &buf_out_histogram);
    clcheck(status);

    /* Define the index space and work-group sizes */
    size_t producer_gbl_sz[2] = { cols, rows };
    size_t producer_loc_sz[2] = { 8, 8 };

    size_t consumer_gbl_sz[1] = { 1 };
    size_t consumer_loc_sz[1] = { 1 };

    status = clEnqueueNDRangeKernel(
            q_gpu,
            kernel_producer,
            2,
            NULL,
            producer_gbl_sz,
            producer_loc_sz,
            0,
            NULL,
            NULL);
    clcheck(status);

#ifndef OCL_PIPES
    /* If pipes aren't supported, we run sequentially */
    clFinish(q_gpu);
#endif

    status = clEnqueueNDRangeKernel(
            has_cpu_dev ? q_cpu : q_gpu,
            kernel_consumer,
            1,
            NULL,
            consumer_gbl_sz,
            consumer_loc_sz,
            0,
            NULL,
            NULL);
    clcheck(status);

    status = clEnqueueReadBuffer(
            has_cpu_dev ? q_cpu : q_gpu,
            buf_out_histogram,
            CL_TRUE,
            0,
            histogram_sz,
            h_output_histogram,
            0,
            NULL,
            NULL);
    clcheck(status);

    float *ref_convolution = convolutionGoldFloat(
            h_input_bmp,
            rows,
            cols,
            filter,
            filter_width);
    int *ref_histogram = histogramGoldFloat(
            ref_convolution,
            num_pixels,
            HIST_BINS);
    int i;
    //for(i=0; i<HIST_BINS; i++) {
    //    printf("%3d histogram(reference): %10d : %10d\n",
    //            i,
    //            h_output_histogram[i],
    //            ref_histogram[i]);
    //}
    int passed = 1;
    for(i=0; i<HIST_BINS; i++) {
        if(h_output_histogram[i] != ref_histogram[i]) {
            printf("cl_computed[%d] = %d != reference[%d] = %d\n",
                    i, h_output_histogram[i],
                    i, ref_histogram[i]);
            passed = 0;
            break;
        }
    }
    if(passed) {
        printf("OpenCL computed convolution/histogram == "
               "reference histogram\n");
    }

    free(ref_histogram);
    free(ref_convolution);
    free(source);

    clReleaseKernel(kernel_consumer);
    clReleaseKernel(kernel_producer);
    clReleaseProgram(program);
    if(has_cpu_dev) {
        clReleaseCommandQueue(q_cpu);
    }
    clReleaseCommandQueue(q_gpu);
    clReleaseMemObject(buf_filter);
    clReleaseMemObject(pipe);
    clReleaseMemObject(img_in);
    clReleaseMemObject(buf_out_histogram);
    clReleaseContext(ctx);

    free(h_input_bmp);
    free(h_output_histogram);

    return EXIT_SUCCESS;
}
