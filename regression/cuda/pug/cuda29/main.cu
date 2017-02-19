#include <call_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 64
#define GRIDSIZE 16
#define NUMBER 6
#define M 128
#define N 256

__host__ void outer_compute(int *A, int *B, int *C);

__host__ void printMatrix(int *A, FILE *fp)
{      for(int i=0;i<N;i++)
  {
      for(int j=0;j<M;j++)
          fprintf(fp,"%d\t", A[i*M+j]);
      fprintf(fp,"\n");
  }
  fprintf(fp,"\n");
}

int main(int argc, char **argv)
{
  int* matrixA;
  int* matrixB;
  int* matrixC;
  int* matrixCPU;

  /* initialization */
  matrixA = (int *) malloc(M*N*sizeof(int));
  matrixB = (int *) malloc(M*N*sizeof(int));
  matrixC = (int *) malloc(M*N*sizeof(int));
  matrixCPU = (int *) malloc(M*N*sizeof(int));

  for (int i=0; i<M; i++) {
	for (int j=0; j<N; j++) {
		matrixA[i*N+j] = rand()%10;
		matrixB[i*N+j] = rand()%10;
		matrixC[i*N+j] = 0;
	}
  }

	/* compute matrix addition */
	outer_compute(matrixA, matrixB, matrixC);

	// Print output to file

	FILE *pFile;
	pFile = fopen ("results.txt" , "w");
	fprintf(pFile,"MATRIX A\n--------\n\n");
	printMatrix(matrixA, pFile);
	fprintf(pFile,"\n\nMATRIX B\n--------\n\n");
	printMatrix(matrixB, pFile);
	fprintf(pFile,"\n\nMATRIX C\n--------\n\n");
	printMatrix(matrixC, pFile);
	fclose (pFile);

	// CPU check

	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			int index=i*N+j;
			matrixCPU[index] = matrixA[index] + matrixB[index];
		}
	}

	int count=0;
	for (int i=0; i<32768; i++) {
		assert(matrixCPU[i] == matrixC[i]);
		if (matrixCPU[i]!=matrixC[i])
			count++;
	}

	printf("\nCPU reports %d mismatches\n", count);
	CUT_EXIT(argc, argv);
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *A,int *B, int *C) {

	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(4,4); // N and M are NOT same so you need to change this

	int index;
	for (int temp=0; temp<32; temp++) {
		int j=64*blockIdx.y+threadIdx.x;
		int i=32*blockIdx.x+temp;
		if( i <M && j <N) {
			index = i*N + j;
			C[index]=A[index]+B[index];
		}
	}
}

__host__ void outer_compute(int *h_matrixA, int *h_matrixB, int *h_matrixC) {
  int *d_matrixA, *d_matrixB, *d_matrixC;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_matrixA,M*N*sizeof(int));
  cudaMalloc((void **) &d_matrixB,M*N*sizeof(int));
  cudaMalloc((void **) &d_matrixC,M*N*sizeof(int));
  cudaMemcpy(d_matrixA,h_matrixA,M*N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB,h_matrixB,M*N*sizeof(int),cudaMemcpyHostToDevice);

  dim3 dimGrid(4,4);
  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<dimGrid,BLOCKSIZE,0*sizeof(int)>>>(d_matrixA,d_matrixB,d_matrixC);

  cudaMemcpy(h_matrixC,d_matrixC,M*N*sizeof(int),cudaMemcpyDeviceToHost);
}

