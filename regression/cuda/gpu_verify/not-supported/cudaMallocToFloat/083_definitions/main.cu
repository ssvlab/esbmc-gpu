#include <call_kernel.h>
//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sm_atomic_functions.h>
#include <stdio.h>

#define N 1 //1024

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
	assert(*D == 0.0);
	//assert(*D == 0.0 || *D == 5);
	assert(*D == 5.0);
/**/	atomicAdd(A,10);
	atomicSub(A,10);
	atomicExch(A,10);
	atomicMin(A,10);
	atomicMax(A,10);
	atomicAnd(A,10);
	atomicOr(A,10);
	atomicXor(A,10);
  	atomicCAS(A,10,11);

/**/	atomicAdd(B,10);
	atomicSub(B,10);
	atomicExch(B,10);
	atomicMin(B,10);
	atomicMax(B,10);
	atomicAnd(B,10);
	atomicOr(B,10);
	atomicXor(B,10);
	atomicInc(B,10);
	atomicDec(B,10);
  	atomicCAS(B,10,11);

/**/	atomicAdd(C,10);
	atomicExch(C,10);
	atomicMin(C,10);
	atomicMax(C,10);
	atomicAnd(C,10);
	atomicOr(C,10);
	atomicXor(C,10);
  	atomicCAS(C,10,11);

	atomicAdd(D,10.0);
	atomicExch(D,10.0);
}

int main (){

	int a = 5;
	int *dev_a;

	cudaMalloc ((void**) &dev_a, sizeof(int));

	cudaMemcpy(dev_a, &a, sizeof(int),cudaMemcpyHostToDevice);

	unsigned int b = 5;
	unsigned int *dev_b;

	cudaMalloc ((void**) &dev_b, sizeof(unsigned int));

	cudaMemcpy(dev_b, &b, sizeof(unsigned int),cudaMemcpyHostToDevice);

	unsigned long long int c = 5;
	unsigned long long int *dev_c;

	cudaMalloc ((void**) &dev_c, sizeof(unsigned long long int));

	cudaMemcpy(dev_c, &c, sizeof(unsigned long long int),cudaMemcpyHostToDevice);
/**/
	float d = 5;
	assert(d==5);
	float *dev_d;// = (float*)malloc (sizeof(float));
	dev_d = (float*)malloc (sizeof(float));

	//cudaMalloc ((void**) &dev_d, sizeof(float));

//	cudaMemcpy(dev_d, &d, sizeof(float),cudaMemcpyHostToDevice);
	memcpy(dev_d,&d, sizeof(float));

	assert (*dev_d == 0);
		//definitions <<<1,N>>>(dev_a,dev_b,dev_c,dev_d);
		ESBMC_verify_kernel_four(definitions,1,N,dev_a,dev_b,dev_c,dev_d);
		
/**/	cudaMemcpy(&a,dev_a,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&b,dev_b,sizeof(unsigned int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&c,dev_c,sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&d,dev_d,sizeof(float),cudaMemcpyDeviceToHost);

/**/	printf("A: %d\n", a); assert(a == 0 || a == 11);
	printf("B: %u\n", b); assert(b == 0 || b == 11);
	printf("C: %u\n", c); assert(c == 0 || c == 11);
	//printf("D: %f\n", d); assert(d == 10.0f || d == 15.0f || d == 5.0f || d == 35.0f);

/**/	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	return 0;
}
