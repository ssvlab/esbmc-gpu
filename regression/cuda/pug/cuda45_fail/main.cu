#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0;
	//srand(time(NULL));
  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }

	//int testsum = 0;
  //for (int i=0; i<SIZE; i++) {
	//	if(in_array[i] == 6)
	//		testsum++;
  //}

  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  //for (int i=0; i<BLOCKSIZE; i++) {
	//	printf("i = %d with val %d \n",i,out_array[i]);
  //}
  sum = out_array[0];

  //printf("Actual SUM = %d == %d CUDA sum \n" , testsum, sum);


  printf ("The number 6 appears %d times in array of  %d numbers\n",sum,SIZE);
  assert(sum == 0);
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

	for(i = 2; i <= BLOCKSIZE; i*=2)
	{
		if(threadIdx.x % i == 0)
		{
			d_out[threadIdx.x] += d_out[threadIdx.x + (i/2)];
		}
		__syncthreads();
	}
	//Or unrolled below... for speed
/*
	if(threadIdx.x % 2 == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + 1];
	}

	__syncthreads();

	if(threadIdx.x % 4 == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + 2];
	}

	__syncthreads();

	if(threadIdx.x % 8 == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + 4];
	}

	__syncthreads();

	if(threadIdx.x % 16 == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + 8];
	}

	__syncthreads();

	if(threadIdx.x % 32 == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + 16];
	}

	//__syncthreads();
  */
}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

