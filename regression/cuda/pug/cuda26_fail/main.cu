#include <call_kernel.h>
#define BLOCKS_PER_GRID_ROW 4
#define BLOCKS_PER_GRID_COLUMN 4
#define ELEMENTS_PER_THREAD 32
#define THREADS_PER_BLOCK 64

#define MATRIX_WIDTH (BLOCKS_PER_GRID_ROW * THREADS_PER_BLOCK)
#define MATRIX_HEIGHT (BLOCKS_PER_GRID_COLUMN * ELEMENTS_PER_THREAD)

#define SIZE (BLOCKS_PER_GRID_ROW * BLOCKS_PER_GRID_COLUMN * ELEMENTS_PER_THREAD * THREADS_PER_BLOCK)

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

__global__ void add_matrix_gpu(int *a, int *b, int *c);

__host__ void add_matrix(int *a, int *b, int *c)
{
  int *dAmatrix, *dBmatrix, *dCmatrix;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &dAmatrix,SIZE*sizeof(int));
  cudaMalloc((void **) &dBmatrix,SIZE*sizeof(int));
  cudaMalloc((void **) &dCmatrix,SIZE*sizeof(int));
  cudaMemcpy(dAmatrix,a,SIZE*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(dBmatrix,b,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  dim3 dimBlock(64, 1);
  dim3 dimGrid(4,4);

  add_matrix_gpu<<<dimGrid,dimBlock>>>(dAmatrix,dBmatrix,dCmatrix);

  /* copy dCmatrix back to c */
  cudaMemcpy(c,dCmatrix,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

}

__global__ void add_matrix_gpu(int *a, int *b, int *c)
{
  int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  int j = blockIdx.y * ELEMENTS_PER_THREAD;
  int index = i+j*MATRIX_WIDTH;
  int count;
  for(count = 0; count < ELEMENTS_PER_THREAD; count++) {
    c[index] = a[index] + b[index];
    index += MATRIX_WIDTH;
  }
}

void print_matrix(int *mat) {
  int i;
  for(i = 1; i < SIZE; i++) {
    printf("%-3d", mat[i]);
    if(i % MATRIX_WIDTH == 0)
      printf("\n");
  }
  printf("\n");
}

int main() {

  int *Amatrix, *Bmatrix, *Cmatrix;

  /* initialization */
  Amatrix = (int *) malloc(SIZE*sizeof(int));
  Bmatrix = (int *) malloc(SIZE*sizeof(int));
  Cmatrix = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    Amatrix[i] = rand()%10;
    Bmatrix[i] = rand()%10;
    Cmatrix[i] = 0;
  }

  printf("A matrix\n");
  print_matrix(Amatrix);
  printf("B matrix\n");
  print_matrix(Bmatrix);

  add_matrix(Amatrix, Bmatrix, Cmatrix);

  printf("C matrix\n");
  print_matrix(Cmatrix);

  for (int i=0; i<SIZE; i++) {
	  assert(Cmatrix[i] == 0);
  }

  free(Amatrix);
  free(Bmatrix);
  free(Cmatrix);

  return 0;
}

