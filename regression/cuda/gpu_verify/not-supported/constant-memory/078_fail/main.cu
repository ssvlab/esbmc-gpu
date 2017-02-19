//xfail:BOOGIE_ERROR
//--blockDim=8 --gridDim=1 --no-inline

// The statically given values for A are not preserved when we translate CUDA
// since the host is free to change the contents of A.
// cf. testsuite/OpenCL/globalarray/pass2

#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 2//8
#define THREAD_CHANGE 1


__constant__ float A[8] = {0,1,2,3,4,5,6,7};

__global__ void globalarray(float* p) {
  int i = threadIdx.x;
  A[THREAD_CHANGE] = 0;		// forçando a entrada no laço, alterando uma constante!
  int a = A[i];

  if(a != threadIdx.x) {
    p[0] = threadIdx.x;	  //entra aqui apenas para para thread=1, por isso não há corrida de dados
  }
}

int main(){

	float *a;
	float *c;
	float *dev_a;
	int size = N*sizeof(float);

	cudaMalloc((void**)&dev_a, size);	

	a = (float*)malloc(size);
	c = (float*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 5;

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);	

//	printf("a:  ");
//	for (int i = 0; i < N; i++)
//		printf("%f	", a[i]);

//	globalarray<<<1,N>>>(dev_a);	
	ESBMC_verify_kernel(globalarray,1,N,dev_a);

	cudaMemcpy(c,dev_a,size,cudaMemcpyDeviceToHost);
	
	//assert(c[0]!=THREAD_CHANGE);		//forçar o ERRO

	free(a);
	free(c);
	cudaFree(dev_a);

	return 0;
}
