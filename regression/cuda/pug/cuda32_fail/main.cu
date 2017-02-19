#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>

// BLOCKS_IN_GRID should be a perfect square
#define BLOCKS_IN_GRID 16
#define THREADS_IN_BLOCK 64
#define WORK_PER_THREAD 32
#define SIZE (WORK_PER_THREAD*THREADS_IN_BLOCK*BLOCKS_IN_GRID)

//debug/verification code
__host__ void print_array(int* a, int size)
{
  printf("[");
  for(int i = 0; i < size-1; i++)
    printf("%2d,", a[i]);
  printf("%2d]\n", a[size-1]);
}

__global__ void add_matrix_gpu(int *a, int *b, int *c, int N)
{
  // Find the index in the array of the first element for this thread
  int row = blockIdx.y * WORK_PER_THREAD;
  int column = blockIdx.x * THREADS_IN_BLOCK + threadIdx.x;
  int rowlen = blockDim.x * gridDim.x;
  int start = row*rowlen + column;

  // Do the addition for this portion of the matrix
  int index = start;
  for(int i = 0; i < WORK_PER_THREAD; i ++)
  {
    c[index] = a[index] + b[index];
    index += rowlen;
  }
}

__host__ int main()
{
  // Initialize the matrices
  int *a,*b,*c;
  a = (int *) malloc(SIZE*sizeof(int));
  b = (int *) malloc(SIZE*sizeof(int));
  c = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    a[i] = rand()%10;
    b[i] = rand()%10;
    c[i] = 0;
  }

  // Set up the grid and blocks
  int gridEdge = sqrtf(BLOCKS_IN_GRID);
  dim3 dimBlock(THREADS_IN_BLOCK);
  dim3 dimGrid(gridEdge, gridEdge);

  // Get data to GPU and go!
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, SIZE*sizeof(int));
  cudaMalloc((void **) &d_b, SIZE*sizeof(int));
  cudaMalloc((void **) &d_c, SIZE*sizeof(int));
  cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  add_matrix_gpu<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, SIZE);

  cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  //debug/verification code
  printf("a= ");
  print_array(a, SIZE);
  printf("\nb= ");
  print_array(b, SIZE);
  printf("\nc= ");
  print_array(c, SIZE);
  //for(int i = 0; i < SIZE; i++)
  //  if(c[i] != a[i] + b[i])
  //    printf("error at %d\n", i);
  for(int i = 0; i < SIZE; i++)
	  assert(c[i] == 0);
}

