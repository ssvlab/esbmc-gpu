//xfail:BOOGIE_ERROR
//possible attempt to modify constant memory
//You can modify the values of the constants, uncomment the lines 14 and 16 to analyze this case.

#include <call_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#define N 2//1024

__constant__ int A[N] = {0, 1, 2, 3};

__global__ void foo(int *B) {

//	assert(A[0]==0);
	A[threadIdx.x] = B[threadIdx.x];
//	assert(A[0]==0); // the constant memory was modified!!!
	__syncthreads();

	B[threadIdx.x] = A[threadIdx.x];

}

int main(){

	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 1;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	//foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nThe function results:\n   ");

	for (int i = 0; i < N; i++){
		printf("%d	", a[i]);
	//		assert(a[i]==i);
	}

	free(a);

	cudaFree(dev_a);

	return 0;
}
