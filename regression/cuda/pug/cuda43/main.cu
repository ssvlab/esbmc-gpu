#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

/* Size of the input random array. */
#define SIZE 256
/* The number of threads that are used to sum. */
#define BLOCKSIZE 32

/* Prototypes for all of the functions. */
__host__ void outer_compute(int *in_arr, int *out_arr);
__device__ int compare(int a, int b);
__global__ void compute(int *d_in,int *d_out);
__host__ void outer_compute(int *h_in_array, int *h_out_array);
/*

 * main
 * This is the main function that initializes the device, sets up the random
 * array of integers, and dispatches to the device to sum the number of 6's.
 */
int main(int argc, char **argv)
{
    int *in_array;
    int i, sum;
    int sumCPU=0;

    /* Initializes the CUDA device. */
    CUT_DEVICE_INIT(argc, argv);

    /* This initializes the input random array. */
    in_array = (int *) malloc(SIZE * sizeof(int));
    if (in_array == NULL)
    {
        fprintf(stderr, "Allocation of array memory failed!");
        return 1;
    }
    for (i = 0; i < SIZE; i++)
    {
        in_array[i] = rand() % 10;
        if (in_array[i] == 6)
        	sumCPU++;
        printf("in_array[%d] = %d\n", i, in_array[i]);
    }

    /* Compute the number of 6's in the array, using the GPU. */
    outer_compute(in_array, &sum);

    /* Print the result. */
    printf("The number 6 appears %d times in array of %d numbers\n", sum, SIZE);

    assert(sum == sumCPU);

    /* Free the array's memory. */
    free(in_array);

    /* Clean up and exit. */
    CUT_EXIT(argc, argv);
    return 0;
}

/*
 * outer_compute
 * This function initializes the memory on the device, copies the input array,
 * calls the 6 counting function, and copies the result back in to the h_sum
 * variable.
 */
__host__ void outer_compute(int *h_in_array, int *h_sum)
{
    int *d_in_array, *d_out_array;

    /* Allocate memory for device copies, and copy input to device. */
    CUDA_SAFE_CALL( cudaMalloc((void **) &d_in_array, SIZE * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc((void **) &d_out_array, BLOCKSIZE * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpy(d_in_array, h_in_array, SIZE * sizeof(int), cudaMemcpyHostToDevice) );

    /* Compute number of appearances of 6's for subset of data in each thread! */
    compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array, d_out_array);
    CUT_CHECK_ERROR("The GPU computation function failed!\n");

    /* Copy the computed integer back in to h_sum. */
    CUDA_SAFE_CALL( cudaMemcpy(h_sum, d_out_array, sizeof(int), cudaMemcpyDeviceToHost) );

    /* Free the memory on the device. */
    CUDA_SAFE_CALL( cudaFree(d_in_array) );
    CUDA_SAFE_CALL( cudaFree(d_out_array) );
}

/*
 * compute
 * This function is ESBMC_executed on the device. It uses BLOCKSIZE threads to sum
 * the number of 6's in the input array. It then reduces this sum in parallel
 * so the final sum will be stored in d_out[0] at the end of the function call.
 */
__global__ void compute(int *d_in, int *d_out)
{
    int i;

    /* Each thread calculates their sum. */
    d_out[threadIdx.x] = 0;
    for (i = 0; i < SIZE/BLOCKSIZE; i++)
    {
        d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
    }

    /* Use the nodes in parallel to reduce the answer to the final sum. */
    for (i = 1; i < BLOCKSIZE; i *= 2)
    {
//        __syncthreads();
        if (threadIdx.x % (i*2) == 0)
        {
            d_out[threadIdx.x] += d_out[threadIdx.x + i];
        }
    }
}

/*
 * compare
 * A simple compare function that ESBMC_executes on the device. It will return 1 if
 * a == b, otherwise 0 is returned.
 */
__device__ int compare(int a, int b)
{
    if (a == b)
        return 1;
    else
        return 0;
}

