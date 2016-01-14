__kernel
void convolution(
        __read_only  image2d_t  img_in,
        __write_only image2d_t  img_out,
              __constant float *filter,
                           int  filter_width,
                     sampler_t  sampler)
{
    /* output (row, col) is specified by global ID of work-item */
    int out_col = get_global_id(0);
    int out_row = get_global_id(1);

    /* calculate filter 'half_width' for indexing the image later */
    int half_width = (int)(filter_width / 2);

    /* initialize 'sum' to contain the accumulated pixel value based on the
     * provided filter */
    float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f };

    int filter_idx = 0;

    /* Coordinates for accessing the input image */
    int2 coords;

    for(int i = -half_width; i <= half_width; i++) {
        coords.y = out_row + i;

        for(int j = -half_width; j <= half_width; j++) {
            coords.x = out_col + j;

            /* Read a pixel from the image. A single-channel image stores the
             * pixel in the 'x' coordinate of the returned vector */
            float4 pixel = read_imagef(img_in, sampler, coords);

            sum.x += pixel.x * filter[filter_idx++];
        }
    }

    /* Copy to output image */
    coords.x = out_col;
    coords.y = out_row;

    write_imagef(img_out, coords, sum);
}
