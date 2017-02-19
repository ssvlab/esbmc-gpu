#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ int outer_compute(int *in_arr);

int main(int argc, char **argv)
{
  int *in_array;
  int sum=0, count=0;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n", i, in_array[i]);
    if (in_array[i] == 6)
    	count++;
  }

  /* compute number of appearances of 6 */
  sum = outer_compute(in_array);

  printf ("The number 6 appears %d times in array of %d numbers\n", sum, SIZE);
  assert(sum != count);
  getchar();
}

__device__ int compare(int a, int b) {
  return (a == b) ? 1 : 0;
}

__device__ __shared__ int localSums[BLOCKSIZE];

__global__ void compute(int *d_in, int* d_sum) {
  int i,stride;
  int localSum = 0;

  for(i=0; i < SIZE/BLOCKSIZE; i++) {
    localSum += compare(d_in[i*BLOCKSIZE+threadIdx.x], 6);
  }
  localSums[threadIdx.x] = localSum;

  // Wait for all threads to finish computation
  __syncthreads();

  // Gather the results in a tree-like fashion
  for(stride = 1; stride < BLOCKSIZE; stride*=2)
  {
    if((threadIdx.x%(stride*2)) == 0)
    {
      localSums[threadIdx.x] += localSums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Put the final result in the sum variable
  if(threadIdx.x == 0)
  {
    d_sum[0] = localSums[0];
  }
}

__host__ int outer_compute(int *h_in_array) {
  int *d_in_array, *d_sum;
  int h_out;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array, SIZE*sizeof(int));
  cudaMalloc((void **) &d_sum, sizeof(int));
  cudaMemcpy(d_in_array, h_in_array, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<1, BLOCKSIZE, (SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array, d_sum);

  cudaMemcpy(&h_out, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
  return h_out;
}

