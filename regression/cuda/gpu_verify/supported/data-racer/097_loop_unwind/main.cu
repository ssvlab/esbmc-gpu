//xfail:BOOGIE_ERROR
//--blockDim=512 --gridDim=64 --loop-unwind=2 --no-inline
//kernel.cu: error: possible write-write race on B

#include <call_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#define N 2//512

extern "C" {

__global__ void helloCUDA(float *A)
{
    __shared__ float B[256];

    for(int i = 0; i < N*2; i ++) {
        B[i] = A[i];
    }
}

}

int main() {

	float *A;
	float *dev_A;

	float size= N*sizeof(float);

	A=(float*)malloc(size);

	cudaMalloc((void**)&dev_A, size);

	for (int i = 0; i < N; i++)
		A[i] = 5;


	cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
	
//		helloCUDA<<<64,N>>>(dev_A);
	ESBMC_verify_kernel(helloCUDA, 1, N, dev_A);

	cudaMemcpy(A, dev_A, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_A);
	free(A);

}
