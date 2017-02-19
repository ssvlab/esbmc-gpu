#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define BLOCK_WIDTH 64
#define BLOCK_HEIGHT 32
#define GRID_DIMENSION 4
#define BLOCK_SIZE BLOCK_WIDTH*BLOCK_HEIGHT
#define MATRIX_SIZE BLOCK_SIZE*GRID_DIMENSION*GRID_DIMENSION

//#define SIZE 256
//#define BLOCKSIZE 32
//#define SIZE 16
//#define BLOCKSIZE 4

__host__ void outer_compute(int *matrix_a, int *matrix_b, int *output_matrix);

int main(int argc, char **argv)
{
	int *matrix_a, *matrix_b, *result_matrix;
	int i, j;


	/* initialization */
	matrix_a = (int*)malloc(sizeof(int)*MATRIX_SIZE);
	matrix_b = (int*)malloc(sizeof(int)*MATRIX_SIZE);
	result_matrix = (int*)malloc(sizeof(int)*MATRIX_SIZE);
	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++)
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			{
				matrix_a[(i*BLOCK_WIDTH*GRID_DIMENSION) + j] = rand()%10;
				matrix_b[(i*BLOCK_WIDTH*GRID_DIMENSION) + j] = rand()%10;
				result_matrix[(i*BLOCK_WIDTH*GRID_DIMENSION) + j] = 0;
			}

	outer_compute(matrix_a, matrix_b, result_matrix);

	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++){
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			assert(result_matrix[(i*BLOCK_WIDTH*GRID_DIMENSION) + j] == 0);
	}

	FILE *outfile;
	outfile = fopen("matrix_output.txt", "w");
	if(!outfile)
		printf("ERROR\n");

	fprintf(outfile, "------Matrix A-------\n");
	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++)
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			fprintf(outfile, "%d ", matrix_a[i*BLOCK_WIDTH*GRID_DIMENSION + j]);
	fprintf(outfile, "\n------Matrix B-------\n");
	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++)
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			fprintf(outfile, "%d ", matrix_b[i*BLOCK_WIDTH*GRID_DIMENSION + j]);
	fprintf(outfile, "\n------Result Matrix-------\n");
	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++)
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			fprintf(outfile, "%d ", result_matrix[i*BLOCK_WIDTH*GRID_DIMENSION + j]);
	fprintf(outfile, "\n");
	for(i=0; i<BLOCK_HEIGHT * GRID_DIMENSION; i++)
		for(j=0; j<BLOCK_WIDTH * GRID_DIMENSION; j++)
			fprintf(outfile, "index = %d, %d + %d = %d\n", i*BLOCK_WIDTH*GRID_DIMENSION + j, matrix_a[i*BLOCK_WIDTH*GRID_DIMENSION + j], matrix_b[i*BLOCK_WIDTH*GRID_DIMENSION + j],
			 result_matrix[i*BLOCK_WIDTH*GRID_DIMENSION + j]);
	fclose(outfile);

	CUT_EXIT(argc, argv);
}


__global__ void myaddMatrices(int *matrix_a, int *matrix_b, int *output_matrix) {

	int increment = BLOCK_WIDTH*GRID_DIMENSION;
	int startAddress = (gridDim.x*blockDim.x*blockIdx.y*BLOCK_HEIGHT) + (blockDim.x*blockIdx.x) + threadIdx.x;
	for(int i=0; i<BLOCK_HEIGHT; i++)
		{
			output_matrix[startAddress + i*increment] = matrix_a[startAddress + i*increment] + matrix_b[startAddress + i*increment];
		}
//	fflush(stdout);
}

__host__ void outer_compute(int *matrix_a, int *matrix_b, int *output_matrix) {

	int *d_matrix_a, *d_matrix_b, *d_output_matrix;
	dim3 dimBlock(BLOCK_WIDTH, 1);
	dim3 dimGrid(GRID_DIMENSION, GRID_DIMENSION);




	cudaMalloc((void **) &d_matrix_a, sizeof(int)*MATRIX_SIZE);
	CUT_CHECK_ERROR("1");
	cudaMalloc((void **) &d_matrix_b, sizeof(int)*MATRIX_SIZE);
	CUT_CHECK_ERROR("2");
	cudaMalloc((void **) &d_output_matrix, sizeof(int)*MATRIX_SIZE);
	CUT_CHECK_ERROR("3");
	cudaMemcpy(d_matrix_a, matrix_a, sizeof(int)*MATRIX_SIZE, cudaMemcpyHostToDevice);
	CUT_CHECK_ERROR("4");
	cudaMemcpy(d_matrix_b, matrix_b, sizeof(int)*MATRIX_SIZE, cudaMemcpyHostToDevice);
	CUT_CHECK_ERROR("5");

	/* compute number of appearances of 8 for subset of data in each thread! */

	//myaddMatrices<<<dimGrid, dimBlock, 0>>> (d_matrix_a, d_matrix_b, d_output_matrix);
	ESBMC_verify_kernel_with_three_args(myaddMatrices, dimBlock, dimBlock, d_matrix_a, d_matrix_b, d_output_matrix);

	cudaMemcpy(output_matrix, d_output_matrix, sizeof(int)*MATRIX_SIZE, cudaMemcpyDeviceToHost);
}

