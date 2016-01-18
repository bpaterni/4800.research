__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE | // pixel-based coordinate system
  CLK_FILTER_NEAREST          | // return nearest pixel to input
  CLK_ADDRESS_CLAMP_TO_EDGE;    // return nearest border pixel

__kernel
void producer(
        image2d_t __read_only    img_in,
#ifdef OCL_PIPES
        pipe __write_only float *pipe_out,
#else
        int                      rows,
        int                      cols,
        __global float          *pipe_out,
#endif
        __constant float*        filter,
        int                      filter_width)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    int half_width = (int)(filter_width / 2);

    float sum = 0.0f;

    int filter_idx = 0;

    int2 coords;

    for(int i=-half_width; i<=half_width; i++) {
        coords.y = row + i;

        for(int j=-half_width; j<=half_width; j++) {
            coords.x = col + j;

            float4 pixel = read_imagef(img_in, sampler, coords);
            sum += pixel.x * filter[filter_idx++];
        }
    }

#ifdef OCL_PIPES
    write_pipe(pipe_out, &sum);
#else
    int gid = row*cols+col;
    pipe_out[gid] = sum;
#endif
}

__kernel
void consumer(
#ifdef OCL_PIPES
        pipe __read_only float *pipe_in,
#else
        __global float         *pipe_in,
#endif
        int                     num_pixels,
        __global int           *histogram)
{
    int pi;
    float pixel;

    for(pi=0; pi<num_pixels; pi++) {
#ifdef OCL_PIPES
        /* Attempt to read from pipe until a pixel becomes available */
        while(read_pipe(pipe_in, &pixel));
#else
        pixel = pipe_in[pi];
#endif

        histogram[(int)pixel]++;
    }
}
