//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=4 --no-inline
//attempt to modify constant memory

#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 2//4

__constant__ int global_constant[N]; //= {0, 1, 2, 3};

__global__ void foo(int *in) {

	global_constant[threadIdx.x] = in[threadIdx.x];

	__syncthreads();

	in[threadIdx.x] = global_constant[threadIdx.x];

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

	//foo<<<1,N>>>(dev_a);
		ESBMC_verify_kernel(foo, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nFunction results:\n   ");

	for (int i = 0; i < N; i++) {
		printf("%d	", a[i]);
	//	assert(a[i] == 0);	//forçar o ERRO
	}

	free(a);	
	
	cudaFree(dev_a);

	return 0;
}
