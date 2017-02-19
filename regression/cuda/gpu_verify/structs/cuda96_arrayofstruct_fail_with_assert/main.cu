#include <call_kernel.h>
//fail
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is race-free.
//
//It uses uses struct-assignment, which is translated into a memcpy by clang and
//dealt with as a series of reads/writes by bugle.

#include <stdio.h>
#include <assert.h>

#define N 2

typedef struct {
  short x;
  short y;
} pair_t;

__global__ void k(pair_t *pairs) {
  pair_t fresh;
  fresh.x = 2; fresh.y = 3;
  pairs[threadIdx.x] = fresh;
}

int main(){
	pair_t *a;
	pair_t *dev_a;
	int size = N*sizeof(pair_t);

	a = (pair_t*)malloc(size);

	/* initialization of a */
	a[0].x = 5; a[1].x = 6;
	a[0].y = 5; a[1].y = 6;

	cudaMalloc((void**)&dev_a, size);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);

	printf("a:\n");
	for (int i = 0; i < N; i++)
		printf("a[%d].x : %d  \ta[%d].y : %d\n", i, a[i].x, i, a[i].y);

	k<<<1,N>>>(dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("new a:\n");
	for (int i = 0; i < N; i++) {
		printf("a[%d].x : %d  \ta[%d].y : %d\n", i, a[i].x, i, a[i].y);
		assert(!(a[i].x == 2  && a[i].y == 3));
	}
	
	free(a);
	
	cudaFree(&dev_a);

	return 0;
}
