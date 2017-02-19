#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N (32 * 4)
#define M (64 * 4)
#define GRIDSIZE 4
#define BLOCKSIZE 64
#define ELEM_TO_OPERATE 32

FILE *fp;

 __global__ void matrixAdd(int *A, int *B, int *C)
{
	int block = blockIdx.x + (blockIdx.y * GRIDSIZE);
	int index = (block * BLOCKSIZE) + threadIdx.x;
	index += (blockIdx.x * 32 );

	for(int i = 0; i < ELEM_TO_OPERATE; i++)
	{
		C[index] = A[index] + B[index];
		//printf ("C[%d]=%d; ", index, C[index]);
		index += M;
	}
	__syncthreads();
}


/* __global__ void matrixAdd(int *A, int *B, int *C)
{
    int x, y;
    int index, end;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * ELEMENTS_PER_THREAD * THREADS_PER_COLUMN + ELEMENTS_PER_THREAD * threadIdx.y;
    index = x + y * MATRIX_WIDTH;

    end = index + ELEMENTS_PER_THREAD * MATRIX_WIDTH;
    for (; index < end; index += MATRIX_WIDTH)
    {
        C[index] = A[index] + B[index];
    }
} */

__host__ void computeMatrixAddition(int *h_A_matrix, int *h_B_matrix, int *h_C_matrix)
{
	int *d_A_matrix, *d_B_matrix, *d_C_matrix;

	cudaMalloc((void **) &d_A_matrix,sizeof(int)* (N*M));
	cudaMalloc((void **) &d_B_matrix,sizeof(int)* (N*M));
	cudaMalloc((void **) &d_C_matrix,sizeof(int)* (N*M));

	cudaMemcpy(d_A_matrix, h_A_matrix, sizeof(int) * N * M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_matrix, h_B_matrix, sizeof(int) * N * M, cudaMemcpyHostToDevice);

	dim3 grid(GRIDSIZE, GRIDSIZE);
	dim3 block(BLOCKSIZE);

	matrixAdd<<<grid,block,0>>>(d_A_matrix, d_B_matrix, d_C_matrix);
	cudaThreadSynchronize();

	cudaMemcpy(h_C_matrix, d_C_matrix, sizeof(int) * N * M, cudaMemcpyDeviceToHost);
}

void printMatrix(int *A, FILE *fp)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<M;j++)
			fprintf(fp,"%d\t", A[i*M+j]);
		fprintf(fp,"\n");
	}

	fprintf(fp,"\n");
}

int main()
{
	int *h_A_matrix, *h_B_matrix, *h_C_matrix;

	fp = fopen("output.txt", "w");

	h_A_matrix = (int *)malloc(sizeof(int) * (N * M));
	h_B_matrix = (int *)malloc(sizeof(int) * (N * M));
	h_C_matrix = (int *)malloc(sizeof(int) * (N * M));

	for(int i=0;i<N;i++)
		for(int j=0;j<M;j++)
		{
			h_A_matrix[i*M+j] = rand() % 9;
			h_B_matrix[i*M+j] = rand() % 9;
		}

	computeMatrixAddition(h_A_matrix,h_B_matrix,h_C_matrix);

	printMatrix(h_A_matrix,fp);
	printMatrix(h_B_matrix,fp);
	printMatrix(h_C_matrix,fp);

	for(int i=0;i<N;i++)
		for(int j=0;j<M;j++)
		{
			printf("%d + ", h_A_matrix[i*M+j]);
			printf("%d = ", h_B_matrix[i*M+j]);
			printf("%d; ", h_C_matrix[i*M+j]);
			assert(h_A_matrix[i*M+j] + h_B_matrix[i*M+j] != h_C_matrix[i*M+j]);
		}

	return 0;

}


