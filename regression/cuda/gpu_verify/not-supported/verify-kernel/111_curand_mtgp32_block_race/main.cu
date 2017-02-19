#include <call_kernel.h>
//fail
//--blockDim=256 --gridDim=1 --no-inline

#include <curand_kernel.h>
//#include <curand_mtgp32_host.h>
#include <stdio.h>

#define N 32 //16

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}

int main() {
	typedef curandStateMtgp32_t tipo;
	float *a;
	float *dev_a;
	tipo *dev_state;
	mtgp32_kernel_params *devKernelParams;

	int size = N*sizeof(float);

	a = (float*)malloc(size);
	cudaMalloc ((void**) &dev_a, size);

	printf("old a:  ");
	for (int i = 0; i < N; i++)
	printf("%f	", a[i]);

	cudaMalloc ( (void**) &dev_state, N*sizeof( tipo ) );

	cudaMalloc((void**)&devKernelParams,sizeof(mtgp32_kernel_params));

	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams); /* Set up constant parameters for the mtgp32 generator */
		/* mtgp32dc_params_fast_11213 is a constant of the type mtgp32_params_fast, it is a system constant */
		/* devKernelParams is the destination*/

	curandMakeMTGP32KernelState(dev_state, mtgp32dc_params_fast_11213, devKernelParams,N, 3412); /* Set up initial states for the mtgp32 generator */
		/*
		 * \param s - pointer to an array of states in device memory
		 * \param params - Pointer to an array of type mtgp32_params_fast_t in host memory
		 * \param k - pointer to a structure of type mtgp32_kernel_params_t in device memory
		 * \param n - number of parameter sets/states to initialize
		 * \param seed - seed value
		 *
		 * */

//	curand_test<<<1,N>>>(dev_state, dev_a);
	ESBMC_verify_kernel(curand_test,1,N,dev_state,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nnew a:  ");
	for (int i = 0; i < N; i++)
		printf("%f	", a[i]);

	free(a);
	cudaFree(&dev_a);
	cudaFree(&dev_state);
	cudaFree(&devKernelParams);

	return 0;
}
