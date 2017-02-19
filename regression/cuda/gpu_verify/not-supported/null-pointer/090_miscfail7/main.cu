#include <call_kernel.h>
//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1
//null pointer access
// ALTOUGH, IT WORKS

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h> 
#define N 2//4//8

__global__ void foo(int *H) {
  size_t tmp = (size_t)H; //type cast
  tmp += sizeof(int);
  int *G = (int *)tmp;
  G -= 1;					//POSSIBLE NULL POINTER ACCESS
  G[threadIdx.x] = threadIdx.x;
  __syncthreads();
  H[threadIdx.x] = G[threadIdx.x];
}

int main() {

	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(N*size);

	for (int i = 0; i < N; i++)
		a[i] = 1;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("a:  ");
	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	//foo<<<1,N>>>(dev_a);
	ESBMC_verify_kernel(foo, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nFunction Results:\n   ");

	for (int i = 0; i < N; i++)
		printf("%d	", a[i]);

	free(a);

	cudaFree(dev_a);

	return 0;
}

