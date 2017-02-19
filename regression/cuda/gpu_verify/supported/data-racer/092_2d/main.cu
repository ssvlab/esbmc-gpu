//data-racer

#include <call_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define SIZE 2
#define TILES 4
#define LENGTH (TILES * SIZE)
#define N 2

__global__ void matrix_transpose(float* A)
{
  __shared__ float tile [SIZE][SIZE];

  int x = threadIdx.x;
  int y = threadIdx.y;

  int tile_x = blockIdx.x;
  int tile_y = blockIdx.y;

	tile[x][y] = A[((x + (tile_x * SIZE)) * LENGTH) + (y + (tile_y * SIZE))];

	tile[x][y] = tile[y][x];

	__syncthreads();

	A[((x + (tile_y * SIZE)) * LENGTH) + (y + (tile_x * SIZE))] = tile[x][y];
}

int main (){
	float *a;
	float *dev_a;
	int size = N*sizeof(float);

	cudaMalloc((void**)&dev_a, size);

	a = (float*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = i;

	cudaMemcpy(dev_a,a,size,cudaMemcpyHostToDevice);

	dim3 GRIDDIM(4,4);
	dim3 BLOCKDIM(2,2);

	//matrix_transpose <<<GRIDDIM, BLOCKDIM>>>(dev_a);
	ESBMC_verify_kernel_f(matrix_transpose, 1, N, dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

	printf("\nResultado de a:\n   ");

	for (int i = 0; i < N; i++){
		printf("%f	", a[i]);
	//	assert(a[i]== (i+1));
	}

	free(a);
	cudaFree(dev_a);
	return 0;
}
