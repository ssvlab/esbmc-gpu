#include <call_kernel.h>
//TEST CASE PASS IN GPU_VERIFY. IT IS NOT VERIFY ARRAY BOUNDS VIOLATION

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#define N 2//64

__global__ void foo(int* p) {
  int* q;
  q = p + 1;
  p[threadIdx.x] = q[threadIdx.x];
}

int main() {
	int *c;
	int *d;
	int *dev_c;

	c = (int*)malloc(N*sizeof(int));
	d = (int*)malloc(N*sizeof(int));

	for (int i = 0; i < N; ++i)
		c[i] = 5;

	cudaMalloc((void**)&dev_c, N*sizeof(int));
	cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

	//foo<<<1, N>>>(dev_c);
	ESBMC_verify_kernel(foo,1,N,dev_c);	

	cudaMemcpy(d, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i){
		assert(d[i]==c[i+1]);
	}

	free(c);
	cudaFree(dev_c);

	return 0;
}
