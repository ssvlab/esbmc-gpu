#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <assert.h>
#define row 256
#define col 128
#define GRID 4
#define BLOCK 16
#define COLTHREAD 32
#define ROWS 128
#define COLUMNS 256
#define THREADS 64

__host__ void printArray(int* A,int* B,int* C){
	int N=32*4;
int M=64*4;
	FILE * pFile;
	pFile = fopen("output.txt","w");

	fprintf(pFile,"A\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",A[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");

	fprintf(pFile,"B\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",B[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");

	fprintf(pFile,"C\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",C[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");
	fclose (pFile);
}

__host__ void outer_compute(int *a, int *b, int *c);

__global__ void add_matrix_gpu(int *a,int *b, int *c)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;

	for(int i=0; i<COLTHREAD; i++){
		int index = COLUMNS*by*COLTHREAD+bx*THREADS+tx+i*COLUMNS;
		c[index] = a[index] + b[index];
		__syncthreads();
	}
}
__host__ void outer_compute(int *h_a_in, int *h_b_in, int *h_c_out) {
  int *d_a_in, *d_b_in, *d_c_out;
	int n=row*col;
	dim3 Block(BLOCK);
	dim3 Grid(GRID,GRID);

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_a_in,n*sizeof(int));
  cudaMalloc((void **) &d_b_in,n*sizeof(int));
  cudaMalloc((void **) &d_c_out,n*sizeof(int));
  cudaMemcpy(d_a_in,h_a_in,n*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_in,h_b_in,n*sizeof(int),cudaMemcpyHostToDevice);

//  add_matrix_gpu<<<Grid,Block>>>(d_a_in,d_b_in,d_c_out);
  ESBMC_verify_kernel_with_three_args(add_matrix_gpu, Grid, Block, d_a_in, d_b_in, d_c_in);
  /*Copy output of device to the host*/
  cudaMemcpy(h_c_out,d_c_out,n*sizeof(int),cudaMemcpyDeviceToHost);
//  printArray(h_a_in,h_b_in,h_c_out,col,row);
}

int main()
{

int *a_matrix, *b_matrix, *c_matrix;

/*INITIALIZATION*/
a_matrix = (int *) malloc((row*col)*sizeof(int ));
b_matrix = (int *) malloc((row*col)*sizeof(int ));
c_matrix = (int *) malloc((row*col)*sizeof(int ));

//Create Matrix A and B
for(int i = 0; i < row; i++)
	{
	for(int j = 0; j < col; j++){
		a_matrix[i*col+j]=rand()%10;
		b_matrix[i*col+j]=rand()%10;
		c_matrix[i*col+j]=0;
	}
}
printf("\n");
//Calculate C Matrix
outer_compute(a_matrix,b_matrix,c_matrix);
printf("\n");
for(int i = 0; i < row; i++)
	{
	for(int j = 0; j < col; j++){
		printf("c_matrix[%d] is %d;\n ",i*col+j,c_matrix[i*col+j]);
		assert(c_matrix[i*col+j] == 0);
	}
}
printf("c_matrix[0] is %d\n",c_matrix[0]);
printArray(a_matrix,b_matrix,c_matrix);
}

