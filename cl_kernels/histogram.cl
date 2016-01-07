#define HIST_BINS 256

__kernel
void histogram(__global int *data,
                        int  num_data,
               __global int *histogram)
{
    __local int local_histogram[HIST_BINS];
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    int sz_loc = get_local_size(0);
    int sz_gbl = get_local_size(0);

    /* Initialize local histogram to zero */
    for(int i=lid;
            i < HIST_BINS;
            i += sz_loc)
    {
        local_histogram[i] = 0;
    }

    /* Wait until all work-items within the work-grouphave completed their
     * stores */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute local histogram */
    for(int i=gid;
            i < num_data;
            i += sz_gbl)
    {
        atomic_add(&local_histogram[data[i]], 1);
    }

    /* Wait until all work-items within the work-group have completed their
     * stores */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Write the local histogram out to the global memory */
    for(int i=lid;
            i < HIST_BINS;
            i += sz_loc)
    {
        atomic_add(&histogram[i], local_histogram[i]);
    }
}
