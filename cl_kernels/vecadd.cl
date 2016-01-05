__kernel
void vecadd(__global int *A,
            __global int *B,
            __global int *C)
{
    // Get the work-item's unique ID
    int gid = get_global_id(0);

    C[gid] = A[gid] + B[gid];
}
