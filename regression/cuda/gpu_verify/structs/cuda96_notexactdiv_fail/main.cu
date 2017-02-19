#include <call_kernel.h>
//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy.
//
//The memcpy destination is unaligned so we have to handle the arrays in and out
//at the byte-level.

#include <stdio.h>

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

#define N 2

typedef struct {
  short x;
  short y;
} s_t; //< sizeof(s_t) == 4

__global__ void k(s_t *in, s_t *out) {
  size_t len = 6;
  memcpy((char*)&out[threadIdx.x]+1, &in[threadIdx.x], len);
}

int main(){
	s_t *a;
	s_t *dev_a;
	s_t *c;
	s_t *dev_c;
	int size = N*sizeof(s_t);

	a = (s_t*)malloc(size);
	c = (s_t*)malloc(size);

	/* initialization of a (the in) */
	a[0].x = 5; a[0].y = 6;
	a[1].x = 5; a[1].y = 6;

	/* initialization of c (the out) */
	c[0].x = 2; c[0].y = 3;
	c[1].x = 2; c[1].y = 3;

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_c, size);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c,c,size, cudaMemcpyHostToDevice);

	printf("a:\n");
	for (int i = 0; i < N; i++)
		printf("a[%d].x : %d  \ta[%d].y : %d\n", i, a[i].x, i, a[i].y);

	printf("c:\n");
	for (int i = 0; i < N; i++)
		printf("c[%d].x : %d  \tc[%d].y : %d\n", i, c[i].x, i, c[i].y);

	k<<<1,N>>>(dev_a, dev_c);

	cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

	printf("new c:\n");
	for (int i = 0; i < N; i++)
		printf("c[%d].x : %d  \tc[%d].y : %d\n", i, c[i].x, i, c[i].y);

	cudaFree(&dev_a);
	cudaFree(&dev_c);

	return 0;
}
