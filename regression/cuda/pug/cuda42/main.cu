#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <device_functions.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

#define DEBUG_MODE false

__host__ void outer_compute(int *in_arr, int *out_arr);


int main(int argc, char **argv)
{
  /* Allow filename as first argument - default to stdout if argument is missing or invalid */
  FILE *file;
  if (argc > 1) {
    file = fopen(argv[1], "w");
    if (file == NULL) {
      fprintf(stderr, "Failed to open file \"%s\" for writing.  Writing to stdout instead.\n", argv[1]);
      file = stdout;
    } else {
      fprintf(stdout, "Writing to file: \"%s\"\n", argv[1]);
    }
  }
  else {
    file = stdout;
  }

  int *in_array, *out_sum;

  /* seed the random number generator */
  srand(1); // ISO-C default seed

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  out_sum = (int *) malloc(sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    out_sum[i] = 0;
    fprintf(file, "in_array[%d] = %d\n",i,in_array[i]);
  }

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_sum);

  fprintf(file, "The number 6 appears %d times in array of  %d numbers\n",*out_sum,SIZE);

  assert(*out_sum == 29);

  /* close the output file */
  if (file != stdout) {
    fclose(file);
  }

  if (DEBUG_MODE) {
    /* Verify results on the CPU */
    int sumcheck = 0;
    for (int i=0; i<SIZE; i++) {
      if (in_array[i] == 6) sumcheck++;
    }
    fprintf(stdout, "CPU SUM VERIFICATION:\nThe number 6 appears %d times in array of  %d numbers\n",sumcheck,SIZE);
  }

}


__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}


__global__ void compute(int *d_in, int *d_out) {
  int i;

  /* Each thread collects count of 6s for its set of numbers */
  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
    d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }

  /* Gather results in parallel */
  int m = 2;
  int logThreads = ceilf(log2f(BLOCKSIZE));
  for (i=0; i<logThreads; i++) {
    /* MUST sync between each level of merging tree values, or race conditions can affect results */
    //__syncthreads();
    if ((threadIdx.x & (m-1)) == 0) { /* when m is a power of 2 (always is here), (i&(m-1)) is equivalent to (i%m), but more efficient */
      /* index to merge with is (m/2) slots away in the array */
      int mergeIdx = threadIdx.x + (m>>1);
      if (mergeIdx < BLOCKSIZE) {
        d_out[threadIdx.x] += d_out[mergeIdx];
      }
    }
    /* Double m for next level of merging */
    m <<= 1;
  }

}


__host__ void outer_compute(int *h_in_array, int *h_out_sum) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 6 for subset of data in each thread and sum in parallel */
  compute<<<1,BLOCKSIZE,0>>>(d_in_array,d_out_array);

  /* copy computed sum back to host */
  cudaMemcpy(h_out_sum,d_out_array,sizeof(int),cudaMemcpyDeviceToHost);
}

