#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <cuda_device_runtime_api.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32
#define KERNELSIZE (SIZE/BLOCKSIZE)

#define CHECK_ERROR() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    fprintf(stderr, "error: %s\n", cudaGetErrorString(err)); \
    exit(1); \
  }}

__device__ int d_arr[SIZE];
__device__ int d_count;
__device__ int teste[SIZE]; /* temporary modification */
__device__ int teste1; /* temporary modification */
__device__ int d_lcount[BLOCKSIZE];

__global__ void compute() {
  int i;
  int si = threadIdx.x*KERNELSIZE;
  int ei = (threadIdx.x+1)*KERNELSIZE;

  d_lcount[threadIdx.x] = 0;
  for(i=si; i<ei; i++)
    if(d_arr[i] == 6) d_lcount[threadIdx.x]++;

  __syncthreads();
  for(i=2; i<=BLOCKSIZE; i*=2) {
    if(!(threadIdx.x % i))
      d_lcount[threadIdx.x] += d_lcount[threadIdx.x+i/2];
    __syncthreads();
  }

  if(!threadIdx.x)
    d_count = d_lcount[threadIdx.x];
}

int main() {
  int i;
  int h_arr[SIZE];
  int h_count;

  printf("contents of random array:\n");
  for(i=0; i<SIZE; i++) {
    h_arr[i] = rand()%10;
    printf("%d ", h_arr[i]);
  }
  printf("\n");

  cudaMemcpyToSymbol(d_arr, &h_arr, SIZE*sizeof(int), 0, cudaMemcpyHostToDevice);
  //CHECK_ERROR() /* temporary comment */
//  compute<<<1,BLOCKSIZE>>>(); /* temporary comment */
  //CHECK_ERROR() /* temporary comment */
  cudaMemcpyFromSymbol(&h_count, &d_count, sizeof(int), 0, cudaMemcpyDeviceToHost); /* temporary comment */
  //cudaMemcpyFromSymbol(&h_count, teste, sizeof(int), 0, cudaMemcpyDeviceToHost); /* temporary modification */
  // cudaMemcpyFromSymbol(&h_count, &teste1, sizeof(int), 0, cudaMemcpyDeviceToHost); /* temporary modification */
  //CHECK_ERROR() /* temporary comment */

  printf ("The number 6 appears %d times in array of  %d numbers\n",h_count,SIZE);

  int h_count_ = 0;
  for(i=0; i<SIZE; i++)
    if(h_arr[i]==6) h_count_++;
  printf("sequential verification, count=%d\n", h_count_);
  assert(h_count == h_count_);
  getchar();

}

