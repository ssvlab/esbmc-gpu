#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 2

__global__ void foo()
{
	__shared__ int A[8];
	A[0] = threadIdx.x;
}

int main(){


	//foo<<<1, N>>>();
	ESBMC_verify_kernel(foo,1, N);		

	cudaThreadSynchronize();

	return 0;
}
