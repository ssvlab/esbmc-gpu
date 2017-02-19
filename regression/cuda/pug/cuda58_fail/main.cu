#include <call_kernel.h>
#include <stdio.h>
//#include <cuda_runtime.h>
//#include <cutil.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32
//#define SIZE 16
//#define BLOCKSIZE 4

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0, count=0;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
    if (in_array[i] == 6)
    	count++;
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

//  for (int i=0; i<BLOCKSIZE; i++) {
//    sum+=out_array[i];
//  }
	// only the 0th element of the array was actually filled from the device
	sum = out_array[0];

  printf ("The number 6 appears %d times in array of %d numbers\n",sum,SIZE);
  assert(sum != count);

//	CUT_EXIT(argc, argv);

	return 0;
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out) {
  int i;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }
  __syncthreads();

	//sum up results
	for (i=2; i<=BLOCKSIZE; i*=2) {
		if (threadIdx.x%i == 0) {
			d_out[threadIdx.x] += d_out[threadIdx.x+i/2];
		}
		__syncthreads();
	}

}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,SIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

	// only copy the first value of the output array here since that's all we need.
  cudaMemcpy(h_out_array,d_out_array,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

