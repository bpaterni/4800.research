__kernel
void
vecadd(__global long *A,
       __global long *B,
       __global long *C)
{
  // Get the work-item's unique ID
  int idx = get_global_id(0);

  // Add the corresponding location of
  // 'A' and 'B', and store the result in 'C'.
  C[idx] = A[idx] + B[idx];
}
