#include <call_kernel.h>
//xfail:BOOGIE_ERROR
//--blockDim=32 --gridDim=64 --no-inline
//error: possible write-write race on

#include <stdio.h>
#include "cuda.h"
#include <assert.h>
#include <cuda_runtime_api.h>

#define M 2//32
#define N 4//64

__global__ void foo(int* p) {
  __shared__ unsigned char x[N];

  for (unsigned int i=0; i<(N/4); i++) {
    ((unsigned int*)x)[i] = 1;//0;
  }
/*
  for (int i = 0; i < N/4; i++) {
	  p[i] = x[i];
  }
*/
}

int main(){
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 2;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

//	foo<<<M,N>>>(dev_a);
	ESBMC_verify_kernel(foo,M,N,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	free(a);

	cudaFree(dev_a);

	return 0;
}
