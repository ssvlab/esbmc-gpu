#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>
#define SIZE 256
#define BLOCKSIZE 32
#define VALUE 6

__host__ void outer_compute (int *in_arr, int *out_arr);

int main (int argc, char **argv)
{
  int *in_array, *out_sum;
  int totalOcc = 0;
  in_array = (int*) malloc (SIZE * sizeof(int));
  out_sum = (int*) malloc (sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    in_array[i] = rand()%10;
  }
  for (int i=0; i<SIZE; i++) {
      printf ("in_array[%d] = %d\n", i, in_array[i]);
      if (in_array[i] == VALUE) {
        totalOcc += 1;
        printf ("Total Occurances till index %d is %d\n", i, totalOcc);
      }
  }
  // initialization
  outer_compute(in_array, out_sum);

   printf ("\nResult = %d\n", *out_sum);
   assert(totalOcc == out_sum[0]);
  //printf ("log2f = %d\n", log2f(SIZE));
  //printf ("log10f = %d\n", log10f(SIZE));
  getchar();
}

__device__ int compare(int a, int b)
{
  if (a == b)
    return 1;
  return 0;
}


__global__ void compute(int *d_in,int *d_out)
{
  d_out[threadIdx.x] = 0;
  for (int i=0; i<SIZE/BLOCKSIZE; i++) {
    int val = d_in[i*BLOCKSIZE + threadIdx.x];
    d_out[threadIdx.x] += compare(val, VALUE);
    //printf("%d", d_out[threadIdx.x]);

  }
  __syncthreads();
}

__global__ void addAllElements(int *d_array_in, int *d_sum_out, int arraySize)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  int numberOfAdditions = log2f (arraySize * 2);
  numberOfAdditions = BLOCKSIZE;
  for (int s = 1; s <= numberOfAdditions/2; s*=2) {
    if ((i % (2*s)) == 0) {
      d_array_in[i] += d_array_in[i + s];
    }
    __syncthreads();
  }
  if (i == 0) d_sum_out[0] = d_array_in[0];
}

__host__ void outer_compute (int *h_in_array, int *h_out_sum)
{
  int *d_in_array, *d_out_array, *d_out_sum;
  int msize;
  cudaMalloc((void **) &d_in_array, SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array, BLOCKSIZE*sizeof(int));
  cudaMalloc((void **) &d_out_sum, sizeof(int));
  cudaMemcpy(d_in_array,
             h_in_array,
             SIZE*sizeof(int),
             cudaMemcpyHostToDevice);

  msize = (SIZE+BLOCKSIZE) * sizeof(int);
  compute<<<1,BLOCKSIZE,msize>>> (d_in_array, d_out_array);

  msize = (1+BLOCKSIZE) * sizeof(int);
  addAllElements<<<1,BLOCKSIZE,msize>>> (d_out_array, d_out_sum, BLOCKSIZE);

  cudaMemcpy(h_out_sum,
             d_out_sum,
             sizeof(int),
             cudaMemcpyDeviceToHost);
}

