#include <call_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

#define DIMX 4*64
#define DIMY 4*32
#define SIZE (DIMX*DIMY)

#define CHECK_ERROR() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    fprintf(stderr, "error: %s\n", cudaGetErrorString(err)); \
    exit(1); \
  }}

__device__ int d_mat_b[SIZE];
__device__ int d_mat_a[SIZE];
__device__ int d_mat_c[SIZE];

int h_mat_b[SIZE];
int h_mat_a[SIZE];
int h_mat_c[SIZE];

__global__ void compute() {
  int i = blockIdx.y * 32 + (blockIdx.x * blockDim.x + threadIdx.x) * DIMY;
  int j;

  for(j=0; j<32; j++,i++)
    d_mat_c[i] = d_mat_a[i] + d_mat_b[i];
}

int main() {
  int i,j;
  dim3 dim_grid(4,4);
  FILE *opf, *sumf;
  int verified;

  opf = fopen("matrix-operands", "w");
  sumf = fopen("matrix-sum", "w");
  if(!opf || !sumf) {
    perror(0);
    exit(1);
  }

  for(i=0; i<SIZE; i++) {
    h_mat_a[i] = h_mat_b[i] = rand()%10;
    h_mat_c[i] = 0;
  }

  for(i=0; i<DIMY; i++) {
    for(j=0; j<DIMX; j++) {
      fprintf(opf, "%d ", h_mat_a[i*DIMX+j]);
    }
    fprintf(opf, "\n");
  }

  cudaMemcpyToSymbol(d_mat_a, &h_mat_a, SIZE*sizeof(int), 0, cudaMemcpyHostToDevice);
  CHECK_ERROR()
  cudaMemcpyToSymbol(d_mat_b, &h_mat_b, SIZE*sizeof(int), 0, cudaMemcpyHostToDevice);
  CHECK_ERROR()
  compute<<<dim_grid, 64>>>();
  CHECK_ERROR()
  cudaMemcpyFromSymbol(&h_mat_c, d_mat_c, SIZE*sizeof(int), 0, cudaMemcpyDeviceToHost);
  CHECK_ERROR()

  for(i=0; i<DIMY; i++) {
    for(j=0; j<DIMX; j++) {
      fprintf(sumf, "%d ", h_mat_c[i*DIMX+j]);
    }
    fprintf(sumf, "\n");
  }

  verified = true;
  for(i=0; i<SIZE; i++) {
    if(h_mat_a[i] + h_mat_b[i] != h_mat_c[i])
      verified = false;
    assert(h_mat_c[i] == 0);
  }
  printf("sequential verification: %s\n", verified?"successful":"failed");
  getchar();
}

