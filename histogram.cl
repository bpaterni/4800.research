#define HIST_BINS 256

__kernel
void histogram(__global int *data,
                        int  num_data,
               __global int *histogram)
{
    __local int loc_hist[HIST_BINS];
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    //printf("gid: %d\n", lid);
    //printf("histogram[%d]: %d\n", lid, histogram[lid]);
    //printf((__constant char *)"%d\n", histogram[lid]);
    //printf((__constant char *)"%d\n", lid);

    /* Initialize local histogram to zero */
    for(int i = lid;
            i < HIST_BINS;
            i += get_local_size(0))
    {
        loc_hist[i] = 0;
    }

    /* Wait until all work-items within the work-group have completed their
     * stores */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute local histogram */
    for(int i = gid;
            i < num_data;
            i += get_global_size(0))
    {
        atomic_add(&loc_hist[data[i]], 1);
    }

    /* Wait until all work-items within the work-group have completed their
     * stores */
    barrier(CLK_LOCAL_MEM_FENCE);

    //printf((__constant char *)"%d\n", loc_hist[lid]);

    /* Write the local histogram out to the global histogram */
    for(int i = lid;
            i < HIST_BINS;
            i += get_local_size(0))
    {
        atomic_add(&histogram[i], loc_hist[i]);
    }
}
