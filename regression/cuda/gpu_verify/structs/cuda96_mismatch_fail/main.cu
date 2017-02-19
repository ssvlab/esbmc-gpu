#include <call_kernel.h>
//fail
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is race-free.
//
//The memcpy is between different src and dst types so we have to handle the
//arrays in and out at the byte-level.

#include <stdio.h>
#include <assert.h>

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)
#define N 2

typedef struct {
  short x;
  char y;
} s1_t; //< sizeof(s1_t) == 4

typedef struct {
  short x;
  short y;
} s2_t; //< sizeof(s2_t) == 4

__global__ void k(s1_t *in, s2_t *out) {
  size_t len = 4;
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}

int main(){
	s1_t *a;
	s1_t *dev_a;
	s2_t *c;
	s2_t *dev_c;
	int size1 = N*sizeof(s1_t);
	int size2 = N*sizeof(s2_t);

	a = (s1_t*)malloc(size1);
	c = (s2_t*)malloc(size2);

	/* initialization of a (the in) */
	a[0].x = 5; a[0].y = 'a';
	a[1].x = 5; a[1].y = 'b';

	/* initialization of c (the out) */
	c[0].x = 2; c[0].y = 3;
	c[1].x = 2; c[1].y = 3;

	cudaMalloc((void**)&dev_a, size1);
	cudaMalloc((void**)&dev_c, size2);

	cudaMemcpy(dev_a,a,size1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c,c,size2, cudaMemcpyHostToDevice);

	printf("a:\n");
	for (int i = 0; i < N; i++)
		printf("a[%d].x : %d  \ta[%d].y : %c\n", i, a[i].x, i, a[i].y);

	printf("c:\n");
	for (int i = 0; i < N; i++)
		printf("c[%d].x : %d  \tc[%d].y : %d\n", i, c[i].x, i, c[i].y);

	k<<<1,N>>>(dev_a, dev_c);

	cudaMemcpy(c,dev_c,size2,cudaMemcpyDeviceToHost);

	printf("new c:\n");
	for (int i = 0; i < N; i++) {
		printf("c[%d].x : %d  \tc[%d].y : %d\n", i, c[i].x, i, c[i].y);
		assert (!(c[i].x = 5 && (c[i].y = 97 || c[i].y = 98)));
	}

	free (a); free (b);
	cudaFree(&dev_a);
	cudaFree(&dev_c);

	return 0;
}

