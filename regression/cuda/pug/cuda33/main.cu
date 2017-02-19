#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>
#define ROWS 128
#define COLUMNS 256
//#define ROWS 10
//#define COLUMNS 10

__host__ void outer_add_matrix_gpu (int *h_in_a_matrix, int *h_in_b_matrix, int *h_out_c_matrix);

int main (int argc, char **argv)
{
  int *a_matrix, *b_matrix, *c_matrix;
  int index;

  a_matrix = (int*) malloc (ROWS * COLUMNS * sizeof(int));
  b_matrix = (int*) malloc (ROWS * COLUMNS * sizeof(int));
  c_matrix = (int*) malloc (ROWS * COLUMNS * sizeof(int));
  //printf ("A & B Matrix = \n");
  for (int iRow = 0; iRow < ROWS; iRow++) {
    //printf ("Row = %d [ ", iRow);
    for (int jColumn = 0; jColumn < COLUMNS; jColumn++) {
      index = iRow * ROWS + jColumn;
	  a_matrix[index] = rand()%10;
	  b_matrix[index] = rand()%10;
	  c_matrix[index] = 0;
	  //printf ("%3d ", a_matrix[index]);
    }
    //printf ("]\n");
  }

  // initialization
  outer_add_matrix_gpu (a_matrix, b_matrix, c_matrix);

  printf ("A + B = C Matrix = \n");
  for (int i = 0; i < ROWS; i++) {
    printf ("Row = %d [ ", i);
    for (int j = 0; j < COLUMNS; j++) {
      index = i * ROWS + j;
      printf ("%3d + %3d = %3d ", a_matrix[index], b_matrix[index], c_matrix[index]);
      assert( c_matrix[index] == b_matrix[index] + a_matrix[index]);
    }
    printf ("]\n");
  }
  getchar();
}

__global__ void add_matrix_gpu(int *a, int *b, int *c)
{
  int iRow = blockIdx.y*blockDim.y + threadIdx.y;
  int jColumn = blockIdx.x*blockDim.x + threadIdx.x;
  int iStartIndex = iRow * 32 * COLUMNS;
  //printf ("iRow, jColumn = (%d, %d)\n", iRow, jColumn);
  //if (iRow % 32 == 0) {
    for (int iRow1 = iRow; iRow1 < 32+iRow; iRow1++) {
      int index = iStartIndex + (iRow1 * COLUMNS) + jColumn;
      if (index < ROWS * COLUMNS) {
	    c[index] = a[index]+ b[index];
      }
    }
  //}
}

__host__ void outer_add_matrix_gpu (int *h_in_a_matrix, int *h_in_b_matrix, int *h_out_c_matrix)
{
  int *d_in_a_matrix, *d_in_b_matrix, *d_out_c_matrix;
  int msize;
  cudaMalloc((void **) &d_in_a_matrix, ROWS * COLUMNS * sizeof(int));
  cudaMalloc((void **) &d_in_b_matrix, ROWS * COLUMNS * sizeof(int));
  cudaMalloc((void **) &d_out_c_matrix, ROWS * COLUMNS * sizeof(int));

  cudaMemcpy(d_in_a_matrix, h_in_a_matrix, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_b_matrix, h_in_b_matrix, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);

  //msize = (ROWS * COLUMNS * 3) * sizeof(int);
  dim3 dimBlock(64);
  dim3 dimGrid(4, 4);
  add_matrix_gpu<<<dimGrid,dimBlock>>>(d_in_a_matrix, d_in_b_matrix, d_out_c_matrix);

  cudaMemcpy(h_out_c_matrix, d_out_c_matrix, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);
}

