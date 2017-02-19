#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define SIZE_ROW 128
#define SIZE_COL 256

#define BLOCKSIZE 64
#define GRIDSIZE_X 4
#define GRIDSIZE_Y 4
#define GRIDSIZE 16
#define ELE_PER_THREAD 32

void printMatrix(int *A, FILE *fp)
{      for(int i=0;i<SIZE_ROW;i++)
   {
       for(int j=0;j<SIZE_COL;j++)
           fprintf(fp,"%d\t", A[i*SIZE_COL+j]);
       fprintf(fp,"\n");
   }
   fprintf(fp,"\n");
}

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
		int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 1) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	//cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}
#endif

__global__ void matrixAdd(int *inMatrix1, int *inMatrix2, int *outMatrix) {
	int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
	for(int i = 0; i < ELE_PER_THREAD; i++) {
		int row = blockIdx.y * ELE_PER_THREAD + i;
		int k = row*SIZE_COL+col;
		*(outMatrix+k) = (*(inMatrix1+k)) + (*(inMatrix2+k));
	}
}


/************************************************************************/
/* Add two matrices                                                       */
/************************************************************************/
int main(int argc, char* argv[])
{
	 Initialize CUDA
	if(!InitCUDA()) {
		return 0;
	}

	// File pointer for the output
	FILE *fp = fopen("matrix_output.txt", "w");

	if(fp == NULL) {
		printf("File could not be open for write.\n");
		CUT_EXIT(argc, argv);
		return -1;
	}

	// Initialize the matrices!
	int *matrix1 = 0, *matrix2 = 0;
	int matrixSize = SIZE_ROW*SIZE_COL;
	matrix1 = (int*)malloc(sizeof(int)*matrixSize);
	matrix2 = (int*)malloc(sizeof(int)*matrixSize);

	if(!matrix1) {
		printf("Matrix 1 could not be allocated!");
		CUT_EXIT(argc, argv);
		return -1;
	}

	if(!matrix2) {
		printf("Matrix 2 could not be allocated!");
		CUT_EXIT(argc, argv);
		return -1;
	}

	for(int i = 0; i < SIZE_ROW; i++) {
		for(int j = 0; j < SIZE_COL; j++) {
			*(matrix1+i*SIZE_COL+j) = rand()%10;
			*(matrix2+i*SIZE_COL+j) = rand()%10;
		}
	}

	fprintf(fp, "Matrix 1\n");
	printMatrix(matrix1, fp);
	fprintf( fp, "\n\n");
	fprintf(fp, "Matrix 2\n");
	printMatrix(matrix2, fp);
	fprintf( fp, "\n\n");

	// Allocate memory in GPU and copy data from matrices
	int *inMatrix1 = 0;
	int *inMatrix2 = 0;
	int *outMatrix = 0;
	CUDA_SAFE_CALL( cudaMalloc((void**) &inMatrix1, sizeof(int)*matrixSize) );
	CUDA_SAFE_CALL( cudaMemcpy(inMatrix1, matrix1, sizeof(int)*matrixSize, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &inMatrix2, sizeof(int)*matrixSize) );
	CUDA_SAFE_CALL( cudaMemcpy(inMatrix2, matrix2, sizeof(int)*matrixSize, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &outMatrix, sizeof(int)*matrixSize) );

	dim3 gridSize(GRIDSIZE_X, GRIDSIZE_Y);

	// Timer for kernel execution
	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer(&timer) );
	CUT_SAFE_CALL( cutStartTimer(timer) );

	// call the matrix add function
	matrixAdd<<<gridSize, BLOCKSIZE, 0>>>(inMatrix1, inMatrix2, outMatrix);
	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutStopTimer(timer) );
	printf( "Processing time (matrixAdd): %f (ms)\n", cutGetTimerValue(timer) );
	CUT_SAFE_CALL( cutDeleteTimer( timer ) );

	// Initialize final matrix
	int *finalMatrix = 0;
	finalMatrix = (int*)malloc(sizeof(int)*matrixSize);
	if(!finalMatrix) {
		printf("finalMatrix could not be allocated!");
		CUT_EXIT(argc, argv);
		return -1;
	}

	// Copy the output from GPU
	CUDA_SAFE_CALL( cudaMemcpy(finalMatrix, outMatrix, sizeof(int)*matrixSize, cudaMemcpyDeviceToHost) );

	fprintf(fp, "Final Matrix\n");
	printMatrix(finalMatrix, fp);
	fprintf( fp, "\n\n");
	fclose(fp);

	int *matrixSum = 0;
	matrixSum = (int*)malloc(sizeof(int)*matrixSize);
	if(!matrixSum) {
		printf("matrixSum could not be allocated! Just assume the final output from GPU is perfect :)");
		CUT_EXIT(argc, argv);
		return -1;
	}

	// Calculate the sum in CPU and comparing with the GPU values
	for(int i = 0; i < SIZE_ROW; i++)
		for(int j = 0; j < SIZE_COL; j++)
			*(matrixSum+i*SIZE_COL+j) = (*(matrix1+i*SIZE_COL+j)) + (*(matrix2+i*SIZE_COL+j));

	printf("Comparing the output with the values calculated in CPU...\n");
	int outputMatches = 1;
	for(int i = 0; i < SIZE_ROW; i++)
		for(int j = 0; j < SIZE_COL; j++)
			if((*(matrixSum+i*SIZE_COL+j)) != (*(finalMatrix+i*SIZE_COL+j))) {
				outputMatches = 0;
				break;
			}

	assert(outputMatches == 1);

	if(outputMatches)
		printf("GPU calculation matches with the CPU!\n");
	else
		printf("GPU calculation does not match with the CPU!\n");

	CUDA_SAFE_CALL( cudaFree(inMatrix1) );
	CUDA_SAFE_CALL( cudaFree(inMatrix2) );
	CUDA_SAFE_CALL( cudaFree(outMatrix) );

	free(matrix1);
	free(matrix2);
	free(finalMatrix);
	free(matrixSum);

	CUT_EXIT(argc, argv);

	return 0;
}

