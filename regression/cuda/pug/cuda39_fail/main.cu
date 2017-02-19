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
	int* in_array;
	int* sum = (int*)malloc(sizeof(int));
	*sum = 0;

	/* Initialization */
	in_array = (int*) malloc(SIZE * sizeof(int));
	for (int i = 0; i < SIZE; i++)
	{
		in_array[i] = rand() % 10;
		printf("in_array[%d] = %d\n", i, in_array[i]);
	}

	/* compute number of appearances of 6 */
	outer_compute(in_array, sum);


	printf ("The number 6 appears %d times in array of  %d numbers\n",*sum,SIZE);
	assert(*sum == 0);
	getchar();
}

__device__ int compare(int a, int b)
{
	if (a == b)
		return 1;
	return 0;
}

__global__ void compute(int *d_in,int *d_out, int* d_sum)
{
	int i;

	d_out[threadIdx.x] = 0;
	for (i = 0; i < SIZE / BLOCKSIZE; i++)
	{
		d_out[threadIdx.x] += compare(d_in[i * BLOCKSIZE + threadIdx.x], 6);
	}

	__syncthreads();

	for(int i = 2, j = 1; i <= BLOCKSIZE; i *= 2, j *= 2)
	{
		if(threadIdx.x % i == 0)
		{
			d_out[threadIdx.x] += d_out[threadIdx.x + j];
		}
		__syncthreads();
	}

	*d_sum = d_out[0];

}

__host__ void outer_compute(int *h_in_array, int *h_sum)
{
	int *d_in_array, *d_sum, *d_out_array;

	/* Allocate memory for device copies, and copy input to device */
	cudaMalloc( (void **) &d_in_array,SIZE * sizeof(int) );
	cudaMalloc( (void **) &d_sum, sizeof(int));
	cudaMalloc( (void **) &d_out_array, BLOCKSIZE * sizeof(int) );
	cudaMemcpy(d_in_array, h_in_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	/* Compute number of appearances of 6 for subset of data in each thread! */
	compute<<<1,BLOCKSIZE,(SIZE + BLOCKSIZE) * sizeof(int)>>>(d_in_array, d_out_array, d_sum);

	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
}

