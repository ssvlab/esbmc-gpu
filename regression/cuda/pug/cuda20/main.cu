#include <call_kernel.h>
#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

//#define LOG_BLOCK_REDUCE_SIZE 6
#define GRIDX             4
#define GRIDY             4
#define BLOCKX            8
#define BLOCKY            8
#define BLOCKC            16
#define THREADC           64
#define NUMS_PER_THREAD   32
//#define THREADS_PER_BLOCK BLOCKX * BLOCKY
#define THREADS_PER_BLOCK THREADC
//#define SRC_ARRAY_SIZE (GRIDX * GRIDY * BLOCKX * BLOCKY * NUMS_PER_THREAD)
#define SRC_ARRAY_SIZE (BLOCKC * THREADC * NUMS_PER_THREAD)

__host__ void array_addition(int *A, int *B, int *C, int *D);

int main(int argc, char **argv)
{
  int *A  = (int *) malloc(SRC_ARRAY_SIZE*sizeof(int));
  int *B  = (int *) malloc(SRC_ARRAY_SIZE*sizeof(int));
  int *C  = (int *) malloc(SRC_ARRAY_SIZE*sizeof(int));
  int *D  = (int *) malloc(SRC_ARRAY_SIZE*sizeof(int));
  memset(A,0,SRC_ARRAY_SIZE*sizeof(int));
  memset(B,0,SRC_ARRAY_SIZE*sizeof(int));
  memset(C,0,SRC_ARRAY_SIZE*sizeof(int));
  memset(D,0,SRC_ARRAY_SIZE*sizeof(int));

  /* array initialization */
  for (int i=0; i<SRC_ARRAY_SIZE; i++) {
    A[i] = rand()%10;
    B[i] = rand()%10;
    D[i] = A[i] + B[i];
    C[i] = 0;
  }

  /* compute number of appearances of 6 */
  array_addition(A, B, C, D);
  getchar();
}

void check_answer(int *A, int *B, int *C, int *D) {
  int diff = 0;
  for (int i=0; i<SRC_ARRAY_SIZE; i++) {
    if ((C[i] != D[i]) || (C[i]!= (A[i]+B[i]))) {
      diff = 1;
	  printf("%3d] = %3d %3d %3d %3d %3d\n", i, A[i], B[i], A[i]+B[i], C[i], D[i]);
    }
  }

  if (diff) {
    printf("Oppps: there was a differenct\n");
  }
  else {
    printf("CPU and GPU arrays are identical\n");
  }
}

void dump_arrays(int *A, int *B, int *C, int *D) {
  printf("[    i] =   A   B A+B   C   D\n");
  for (int i=0; i<SRC_ARRAY_SIZE; i++) {
	printf("[%5d] = %3d %3d %3d %3d %3d\n", i, A[i], B[i], A[i]+B[i], C[i], D[i]);
  }
  printf("[    i] =   A   B A+B   C   D\n");
}

#define BLOCK_SIZE (THREADS_PER_BLOCK * NUMS_PER_THREAD)

__global__ void array_addition_kernel_both_flat(int *A, int *B, int *C) {
  int k;
  int idx = (blockIdx.x * BLOCK_SIZE);
  //sums a column
  for (k=0; k<NUMS_PER_THREAD; k++) {
	  int lidx = idx + k * THREADS_PER_BLOCK + threadIdx.x;
      C[lidx] = A[lidx] + B[lidx];
  }
}
__global__ void array_addition_kernel_flat_threads(int *A, int *B, int *C) {
  int k;
  int idx = ((blockIdx.x * gridDim.x + blockIdx.y)* BLOCK_SIZE);
  //sums a column
  for (k=0; k<NUMS_PER_THREAD; k++) {
	  int lidx = idx + k * THREADS_PER_BLOCK + threadIdx.x;
      C[lidx] = A[lidx] + B[lidx];
  }
}
__global__ void array_addition_kernel_2d_threads(int *A, int *B, int *C) {
  int k;
  int idx =  ((blockIdx.x * gridDim.x + blockIdx.y)* BLOCK_SIZE);
      idx += (threadIdx.x * blockDim.x * NUMS_PER_THREAD);
  for (k=0; k<NUMS_PER_THREAD; k++) {
	  int lidx = idx + (k * blockDim.y) + threadIdx.y;
      C[lidx] = A[lidx] + B[lidx];
  }
}

__host__ void array_addition(int *A, int *B, int *C, int *D) {
  int *d_A;
  int *d_B;
  int *d_C;
  dim3 dimGrid(GRIDX,GRIDY);
  dim3 dimBlock(BLOCKX, BLOCKY);
  dim3 dimBlock2(1, THREADC);

  /* allocate memory for device copies, and copy input to device */
//#define CUT_SAFE_MALLOC /**/
  CUT_SAFE_MALLOC(cudaMalloc((void **) &d_A, SRC_ARRAY_SIZE*sizeof(int)));
  CUT_SAFE_MALLOC(cudaMalloc((void **) &d_B, SRC_ARRAY_SIZE*sizeof(int)));
  CUT_SAFE_MALLOC(cudaMalloc((void **) &d_C, SRC_ARRAY_SIZE*sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_A, A, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_B, B, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));

  /* compute number of appearances of 8 for subset of data in each thread! */

  printf("array_addition_kernel_both_flat  \n");
  array_addition_kernel_both_flat<<<BLOCKC, THREADC>>>(d_A, d_B, d_C);
  CUT_CHECK_ERROR("array_addition_kernel died");
  memset(C, 0, SRC_ARRAY_SIZE*sizeof(int));
  cudaMemcpy(C, d_C, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  check_answer(A, B, C, D);


  printf("array_addition_kernel_flat_threads\n");
  //array_addition_kernel_flat_threads<<<dimGrid, THREADC>>>(d_A, d_B, d_C);
  CUT_CHECK_ERROR("array_addition_kernel died");
  memset(C, 0, SRC_ARRAY_SIZE*sizeof(int));
  cudaMemcpy(C, d_C, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  check_answer(A, B, C, D);


  printf("array_addition_kernel_2d_threads\n");
  //array_addition_kernel_2d_threads<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  CUT_CHECK_ERROR("array_addition_kernel died");
  memset(C, 0, SRC_ARRAY_SIZE*sizeof(int));
  cudaMemcpy(C, d_C, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  check_answer(A, B, C, D);

  printf("array_addition_kernel_2d_threads dumBlock2(1, THREADC)\n");
  //array_addition_kernel_2d_threads<<<dimGrid, dimBlock2>>>(d_A, d_B, d_C);
  CUT_CHECK_ERROR("array_addition_kernel died");
  memset(C, 0, SRC_ARRAY_SIZE*sizeof(int));
  cudaMemcpy(C, d_C, SRC_ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  check_answer(A, B, C, D);

  for (int i=0; i<SRC_ARRAY_SIZE; i++) {
      assert((A[i] + B[i] == D[i])&&(D[i] == C[i]));
  }

  dump_arrays(A, B, C, D);
}

/*
 * vim: syntax=c :
 */

