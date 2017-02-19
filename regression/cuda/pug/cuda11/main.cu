#include <call_kernel.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#define THREADS					8 //64
#define ELEMENTS_PER_THREAD		8//32
#define BLOCK_WIDTH				4
#define BLOCK_HEIGHT			4
#define BLOCK_SIZE				BLOCK_WIDTH * BLOCK_HEIGHT
#define MATRIX_WIDTH			THREADS * BLOCK_WIDTH
#define MATRIX_HEIGHT			ELEMENTS_PER_THREAD * BLOCK_HEIGHT
#define MATRIX_SIZE				MATRIX_WIDTH * MATRIX_HEIGHT

__global__ void add_matrix_gpu(int *matrixA, int *matrixB, int *matrixC)
{
	int i = blockIdx.x * THREADS + threadIdx.x;
	int j = blockIdx.y * ELEMENTS_PER_THREAD;
	int index = i + j * MATRIX_WIDTH;
	for( int k = 0; k < ELEMENTS_PER_THREAD ; k++ )
	{
		matrixC[index] = matrixA[index] + matrixB[index];
		index+=MATRIX_WIDTH;
	}
}

int main(int argc, char **argv)
{
	srand(5);
	int debug = argc > 1;

	int *matrixA, *matrixB, *matrixC;

	matrixA = (int*)malloc( MATRIX_SIZE * sizeof(int) );
	matrixB = (int*)malloc( MATRIX_SIZE * sizeof(int) );
	matrixC = (int*)malloc( MATRIX_SIZE * sizeof(int) );

	for( int i = 0; i < MATRIX_SIZE; i++ )
	{
		matrixA[i] = rand() % 10;
		matrixB[i] = rand() % 10;
		matrixC[i] = 0;
	}

	int *d_matrixA, *d_matrixB, *d_matrixC;

	cudaMalloc( (void **) &d_matrixA, MATRIX_SIZE * sizeof(int) );
	cudaMalloc( (void **) &d_matrixB, MATRIX_SIZE * sizeof(int) );
	cudaMalloc( (void **) &d_matrixC, MATRIX_SIZE * sizeof(int) );
	cudaMemcpy(d_matrixA, matrixA, MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, matrixB, MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, matrixC, MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS, 1);
	dim3 dimGrid(BLOCK_WIDTH, BLOCK_HEIGHT);
	//add_matrix_gpu<<<dimGrid, dimBlock>>>(d_matrixA, d_matrixB, d_matrixC);
	ESBMC_verify_kernel_with_three_args(add_matrix_gpu,dimGrid,dimBlock,d_matrixA, d_matrixB, d_matrixC);

	cudaMemcpy(matrixC, d_matrixC, MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		/*if(matrixA[i] + matrixB[i] == matrixC[i])
			printf("matrixA[%d] = %d\tmatrixb[%d] = %d\tmatrixC[%d] = %d\n", i, matrixA[i], i, matrixB[i], i, matrixC[i]);
		else
			break;*/
		printf("matrixA[%d] = %d\tmatrixb[%d] = %d\tmatrixC[%d] = %d\n", i, matrixA[i], i, matrixB[i], i, matrixC[i]);
		assert(matrixC[i]==matrixA[i]+matrixB[i]);
	}
#ifdef DEBUG
	printf("\nPress Enter To Continue...\n");
	while (1)
	{
		if ('\n' == getchar())
		   break;
	}
#endif
	getchar();
	return 0;
}

