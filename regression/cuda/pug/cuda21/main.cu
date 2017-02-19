#include <call_kernel.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define BLOCKS_PER_GRID 16
#define THREADS 64
#define ELEMENTS_PER_THREAD 32

#define MATRIX_WIDTH THREADS*sqrt(BLOCKS_PER_GRID*1.0)
#define MATRIX_HEIGHT ELEMENTS_PER_THREAD*sqrt(BLOCKS_PER_GRID*1.0)


__host__ void add_matrix(int *h_a_matrix, int *h_b_matrix, int *h_c_matrix);

int main(int argc, char **argv)
{
  int debug = 1;
  if(argc > 1)
	debug = *((int*)argv[1]);
  int *a_matrix = (int*)malloc(MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));
  int *b_matrix = (int*)malloc(MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));
  int *c_matrix = (int*)malloc(MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));


  //srand(151);

  /* initialization */
  int index;
  for (int i=0;i<MATRIX_WIDTH;i++) {
    for (int j=0;j<MATRIX_HEIGHT;j++) {
	  index =i+j*MATRIX_HEIGHT;
	  a_matrix[index]=rand()%10;
	  b_matrix[index]=rand()%10;
	  c_matrix[index]=0;
    }
  }

  /* Add the 2 matrices */
  add_matrix(a_matrix,b_matrix,c_matrix);

  if(debug){
    int *c_matrix_debug = (int*)malloc(MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));

	int i, j, index;
	for (i=0;i<MATRIX_WIDTH;i++) {
	  for (j=0;j<MATRIX_HEIGHT;j++) {
		index =i+j*MATRIX_HEIGHT;
		c_matrix_debug[index]=a_matrix[index]+b_matrix[index];
	  }
    }

    for (i=0;i<MATRIX_WIDTH;i++) {
	  for (j=0;j<MATRIX_HEIGHT;j++) {
		index =i+j*MATRIX_HEIGHT;
		if(c_matrix_debug[index] != c_matrix[index])
			printf("Incorrect entry at %d, %d\nEntry is: %d\nCorrect entry is %d\n", i, j, c_matrix[index], c_matrix_debug[index]);
	  }
    }

  }

  printf("Here is my pretty matrix:\n");
  for (int i=0;i<MATRIX_HEIGHT;i++) {
	  printf("Row:%d",i);
	  for (int j=0;j<MATRIX_WIDTH;j++) {
		index =j+i*MATRIX_HEIGHT;
		printf("\t%d", c_matrix[index]);
		assert(c_matrix[index] == a_matrix[index]+b_matrix[index]);
	  }
	  printf("\n");
  }
  getchar();
}

__global__ void add_matrix_gpu(int *a_matrix, int *b_matrix, int *c_matrix)
{
  int i=blockIdx.x*THREADS+threadIdx.x;
  int j=blockIdx.y*ELEMENTS_PER_THREAD;
  int index = i+j*MATRIX_WIDTH;
  for(int k=0; k < ELEMENTS_PER_THREAD; k++)
  {
	c_matrix[index]=a_matrix[index]+b_matrix[index];
	index+=MATRIX_WIDTH;
  }
}

__host__ void add_matrix(int *h_a_matrix, int *h_b_matrix, int *h_c_matrix) {
  int *d_a_matrix, *d_b_matrix, *d_c_matrix;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_a_matrix, MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));
  cudaMalloc((void **) &d_b_matrix, MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));
  cudaMalloc((void **) &d_c_matrix, MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int));

  cudaMemcpy(d_a_matrix,h_a_matrix,MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_matrix,h_b_matrix,MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int),cudaMemcpyHostToDevice);

  /* Sum up the matrices */
  int gridDimension = (int)sqrt(BLOCKS_PER_GRID*1.0);
  dim3 dimBlock(THREADS,1);
  dim3 dimGrid(gridDimension,gridDimension);

  add_matrix_gpu<<<dimGrid,dimBlock>>>(d_a_matrix,d_b_matrix,d_c_matrix);

  cudaMemcpy(h_c_matrix,d_c_matrix,MATRIX_WIDTH*MATRIX_HEIGHT*sizeof(int),cudaMemcpyDeviceToHost);
}

