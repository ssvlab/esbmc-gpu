#include <call_kernel.h>
//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=2
//assert

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define DIM 2
#define N DIM*DIM

typedef struct {

	unsigned int a, b;

} pair;
__device__ void assertion(pair* A) {
 // __assert(false);
	assert(0);
}

__global__ void test(pair* A)
{
	assertion(A);
}

int main () {

	//It's necessary to set a device variable (dev_a), although it was not used, otherwise the kernel will not be launched.
	pair a;
	pair *dev_a;

	/* initialization of a */
	a.a = 5; a.b = 5;

	cudaMalloc((void**)&dev_a, sizeof(pair));

	cudaMemcpy(dev_a,&a,sizeof(pair), cudaMemcpyHostToDevice);

	printf("old a:\n");
		printf("a.a : %u  \ta.b : %u\n", a.a, a.b);

	test<<<DIM,DIM>>>(dev_a);

	cudaMemcpy(&a,dev_a,sizeof(pair),cudaMemcpyDeviceToHost);

	printf("new a:\n");
		printf("a.a : %u  \ta.b : %u\n", a.a, a.b);
		//assert ((a[i].a == 2 && a[i].b == 3));

	cudaFree(&dev_a);

	return 0;
}
