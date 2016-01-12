__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE | // use pixel-based integer addressing
  CLK_FILTER_LINEAR |           // linear interpolation of surrounding pixels
  CLK_ADDRESS_CLAMP;            // return 'black' Out Of Bounds' coords

__kernel
void rotation(
        __read_only  image2d_t img_in,
        __write_only image2d_t img_out,
                           int img_width,
                           int img_height,
                         float theta)
{
    /* (x,y) is output coordinates, specified by global ID of work-item */
    int x = get_global_id(0);
    int y = get_global_id(1);

    /* compute image center */
    float x0 = img_width  / 2.0f;
    float y0 = img_height / 2.0f;

    /* compute work-item's location relative to the image center */
    int xprime = x-x0;
    int yprime = y-y0;

    float sintheta = sin(theta);
    float costheta = cos(theta);

    /* compute the input location */
    float2 pix_in;
    pix_in.x = xprime*costheta - yprime*sintheta + x0;
    pix_in.y = xprime*sintheta + yprime*costheta + y0;

    /* retrieve pixel value from input image location */
    float value = read_imagef(img_in, sampler, pix_in).x;

    /* write to output image */
    write_imagef(img_out, (int2)(x, y), (float4)(value, 0.f, 0.f, 0.f));
}
