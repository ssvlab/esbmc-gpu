#include <call_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32
#define NUMBER 6
__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int* in_array;
  int* out_array;
  int cpu_out=0;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }

  for (int index=0; index<SIZE; index++)
	if (in_array[index] == NUMBER)
		cpu_out++;

  out_array = (int*) malloc(sizeof(int)*BLOCKSIZE);

  /* compute number of appearances of NUMBER */
  outer_compute(in_array, out_array);

  printf ("The number %d appears %d times in array of %d numbers\n",NUMBER, out_array[0],SIZE);
  printf ("The corresponding CPU output is %d\n",cpu_out);

  assert(out_array[0] != cpu_out);

  CUT_EXIT(argc, argv);
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out) {
  int i,a;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],NUMBER);
  }

  __syncthreads();

	for (a=2, i=BLOCKSIZE; i>1; a*=2, i=i/2) {
		if (threadIdx.x % a == 0) {
			d_out[threadIdx.x] += d_out[threadIdx.x + (a/2)];
			}
			__syncthreads();
		}
}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

