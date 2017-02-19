#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define THREADS_PER_BLOCK 64
#define BLOCKS_PER_GRID 16
#define BLOCKS_IN_ROW 4 //Sqrt(BLOCKS_PER_GRID)
#define ELEMENTS_PER_THREAD 32
#define N THREADS_PER_BLOCK*BLOCKS_PER_GRID*ELEMENTS_PER_THREAD   //total number of elements
#define MATRIX_WIDTH THREADS_PER_BLOCK*BLOCKS_IN_ROW
#define MATRIX_HEIGHT ELEMENTS_PER_THREAD*BLOCKS_IN_ROW

__host__ void start_work(int *a, int *b, int *c);

int main(int argc, char **argv)
{

  /* initialization */
  int *a;
  int *b;
  int *c;
  int debug = 1;
  int print = 1;

  srand(12);

  a = (int*)malloc(N * sizeof(int));
  b = (int*)malloc(N * sizeof(int));
  c = (int*)malloc(N * sizeof(int));

  for(int i=0; i<MATRIX_WIDTH; i++){
	for(int j=0; j<MATRIX_HEIGHT; j++){
		int index = i + j * MATRIX_HEIGHT;
		a[index] = rand() % 10;
		b[index] = rand() % 10;
		c[index] = 0;
	}
  }

  start_work(a, b, c);

  if(debug){
	int same = 1;
	for(int i=0; i<MATRIX_WIDTH; i++){
		for(int j=0; j<MATRIX_HEIGHT; j++){
			int index = i+j * MATRIX_HEIGHT;
			if(a[index] + b[index] != c[index]){
				printf("c[%d] = %d instead of %d\n", index, c[index], a[index]+b[index]);
				same = 0;
			}
		}
	}
	if(!same)
		printf("Something didn't work in the parallel version.\n");
  }

  //Print matrix c
  if(print){
	printf("Matrix C is as follows (printed row by row):\n");
	for(int j=0; j<MATRIX_HEIGHT; j++){
		printf("R %d \t", j);
		for(int i=0; i<MATRIX_WIDTH; i++){
			int index = i+j * MATRIX_HEIGHT;
			printf("%d\t", c[index]);
			assert(c[index] == a[index]+b[index]);
		}
		printf("\n");
	}
  }
}

__global__ void add_matrix_gpu(int *a, int *b, int *c){
	int i = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	int j = blockIdx.y*ELEMENTS_PER_THREAD;
	int index = i+j*MATRIX_WIDTH;
	for(int k=0; k < ELEMENTS_PER_THREAD; k++){
		c[index] = a[index] + b[index];
		index += MATRIX_WIDTH;
    }
}

__host__ void start_work(int *h_a, int *h_b, int *h_c) {
  int *d_a, *d_b, *d_c;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_a, N * sizeof(int));
  cudaMalloc((void **) &d_b, N * sizeof(int));
  cudaMalloc((void **) &d_c, N * sizeof(int));
  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimBlock(THREADS_PER_BLOCK, 1);
  dim3 dimGrid(BLOCKS_IN_ROW, BLOCKS_IN_ROW);

  add_matrix_gpu<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
}

