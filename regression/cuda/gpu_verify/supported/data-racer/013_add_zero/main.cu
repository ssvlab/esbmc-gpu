#include <call_kernel.h>
//fail: data-race, all the threads write on A[0]

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sm_atomic_functions.h>

#define N 2

__global__ void race_test (unsigned int* i, int* A)
{
	int tid = threadIdx.x;
	int j = atomicAdd(i,0);
  	A[j] = tid;
}

  int main(){

  	unsigned int *i;
  	int *A;
  	unsigned int *dev_i;
  	int *dev_A;

  	A = (int*)malloc(N*sizeof(int));

  	for (int t = 0; t < N; ++t){
  		A[t] = 11;
  		printf(" %d  ", A[t]);
  	}

  	i = (unsigned int*)malloc(sizeof(unsigned int));

	*i = 0;
  	
	cudaMalloc((void**)&dev_A, N*sizeof(int));
  	cudaMalloc((void**)&dev_i, sizeof(unsigned int));

  	cudaMemcpy(dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice);
  	cudaMemcpy(dev_i, i, sizeof(unsigned int), cudaMemcpyHostToDevice);

  		//race_test<<<1,N>>>(dev_i, dev_A);
		ESBMC_verify_kernel_u(race_test,1,N,dev_i,dev_A);

  	cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);

  	for (int t=0; t<N;t++){
  		printf("A[%d]=%d; ", t, A[t]);
  	}
	
	//assert(A[0] == 11);
	assert(A[0] == 0 || A[0] == 1); // A[0] == i,where i = [0,N-1]

  	free(A);
  	free(i);
  	cudaFree(dev_A);
  	cudaFree(dev_i);

  	return 0;
}
