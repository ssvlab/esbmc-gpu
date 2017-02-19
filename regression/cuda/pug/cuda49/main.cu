#include <call_kernel.h>
#include <stdio.h>
#include "cuda.h"
#include <cuda_runtime_api.h>
#include <assert.h>

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------

#define BLOCKSIZE 	32
#define NUMBER 		6
#define SIZE		256

// Uncomment the following define to see the program print out the numbers and
// force there to be at least a single instance of NUMBER in the array
#define DEBUG

//-----------------------------------------------------------------------------
// Function prototypes
//-----------------------------------------------------------------------------

__device__	int		compare(int a, int b);
__global__	void	compute(int *d_int, int* d_out, int* sum);
__host__ 	void 	outer_compute(int *in_arr, int *out_arr);

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

__device__ int compare(int a, int b)
{
	if (a == b)
	{
		return 1;
	}

	return 0;
}

__global__ void compute(int* d_in, int* d_out, int* sum)
{
	int i;
	int	n = 2;

	d_out[threadIdx.x] = 0;

	for (i=0; i<SIZE/BLOCKSIZE; i++)
	{
		d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x], NUMBER);


	}
	//printf("%d; ", d_out[threadIdx.x]);
	__syncthreads();
	while (n <= SIZE)
	{
		if ((threadIdx.x % n) == 0)
		{
			if (threadIdx.x + n/2 < BLOCKSIZE)
			{
				d_out[threadIdx.x] += d_out[threadIdx.x + n/2];
				//printf("d_out[%d]=%d; ", threadIdx.x, d_out[threadIdx.x]);
			}
		}

		n = n * 2;
		__syncthreads();
	}
	//printf("d_out[%d]=%d; ", threadIdx.x, d_out[threadIdx.x]);
	__syncthreads();

	if (threadIdx.x == 0)
	{
		(*sum) = d_out[0];
		//printf("%d", sum[threadIdx.x]);
	}
}

__host__ void outer_compute(int* h_in_array, int* h_sum)
{
	int*	d_in_array;
	int*	d_out_array;
	int*	d_sum;

	// Allocate memory for device copies
	cudaMalloc((void **) &d_in_array, SIZE*sizeof(int));
	cudaMalloc((void **) &d_out_array, SIZE*sizeof(int));
	cudaMalloc((void **) &d_sum, sizeof(int));

	// Initialize the sum
	*h_sum = 0;

	// Copy over the array and initialized sum
	cudaMemcpy(d_in_array, h_in_array, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);

	// compute number of appearances of NUMBER for subset of data in each thread!
	compute<<<1,BLOCKSIZE,0>>>(d_in_array, d_out_array, d_sum);

	// Copy the summation back from the device
	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
}

//-----------------------------------------------------------------------------
// Main Function
//-----------------------------------------------------------------------------

int main(int argc, char **argv)
{
#if defined(DEBUG) || defined(_DEBUG)
	int		cpu_count = 0;
#endif

	int*	in_array;
	int 	sum = 0;

	// Seed rand() so we always get the same random numbers for testing purposes
	srand(0);

	// Initialize the input array
	in_array = (int *) malloc(SIZE*sizeof(int));

	for (int i=0; i<SIZE; i++)
	{
		in_array[i] = rand()%10;

#if defined(DEBUG) || defined(_DEBUG)
		if (in_array[i] == NUMBER)
		{
			cpu_count++;
		}
#endif
	}

#if defined(DEBUG) || defined(_DEBUG)
	if (cpu_count == 0)
	{
		in_array[SIZE-1] = NUMBER;
		cpu_count++;
	}

	for (int i = 0; i < SIZE; i++)
	{
		printf("in_array[%d] = %d\n", i, in_array[i]);
	}
#endif

	// Compute number of appearances of NUMBER
	outer_compute(in_array, &sum);
	printf("\n");

	printf ("The number %d appears %d times in array of  %d numbers", NUMBER, sum, SIZE);

#if defined(DEBUG) || defined(_DEBUG)
	printf (" (Should be %d)", cpu_count);
#endif
	assert(sum == cpu_count);
	printf ("\n");
}

