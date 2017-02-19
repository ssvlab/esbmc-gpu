//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: possible null pointer access

#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 2//8

#define tid (blockIdx.x * blockDim.x + threadIdx.x)

__device__ float multiplyByTwo(float *v, unsigned int index)
{
    return v[index] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int index)
{
    return v[index] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foo(float *v)
{
    funcType f = (funcType)3; // it's a null pointer
    v[tid] = (*f)(v, tid);
}

int main(){

	float* w;
	float* dev_w;

	int size = N*sizeof(float);

	w =(float*) malloc(size);

	for (int i = 0; i < N; ++i){
		w[i] = i;
	}


	cudaMalloc((void**)&dev_w, size);

	cudaMemcpy(dev_w,w, size,cudaMemcpyHostToDevice);

	//foo <<<1,N>>>(dev_w);
	ESBMC_verify_kernel_f(foo, 1, N, dev_w);

	cudaMemcpy(w,dev_w,size,cudaMemcpyDeviceToHost);

	printf("\nw:");
	for (int i = 0; i < N; ++i){
		printf(" %f	",	w[i]);
//		assert(!(w[i] == i));
	}

	//printf ("\n (float) functype: %f", divideByTwo)//3.5;

	free(w);
	cudaFree(dev_w);

	return 0;
}
