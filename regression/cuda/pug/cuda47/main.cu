#include <call_kernel.h>
#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define SRC_ARRAY_SIZE 256
#define BLOCKSIZE 32
#define LOG_BLOCKSIZE 5

__host__ void outer_compute(int *h_in_array, int *h_out_array);
__global__ void reduce(int* d_out);

int main(int argc, char **argv)
{
  int *in_array   = (int *) malloc(SRC_ARRAY_SIZE*sizeof(int));
  int *out_array  = (int *) malloc(BLOCKSIZE*sizeof(int));
  int cnt6=0;

  /* initialization */
  for (int i=0; i<SRC_ARRAY_SIZE; i++) {
    in_array[i] = rand()%10;
    out_array[i] = 0;
    if( in_array[i] == 6)
      cnt6++;
    printf("in_array[%d] = %2d %2d\n",i,in_array[i], cnt6);
  }

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  printf ("The number 6 appears %d GPU times and %d CPU times in array of %d numbers\n", out_array[0], cnt6, SRC_ARRAY_SIZE);

  assert(out_array[0] == cnt6);

  getchar();
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in, int *d_out) {
  int i;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SRC_ARRAY_SIZE/BLOCKSIZE; i++) {
      //d_out[threadIdx.x] += ( (d_in[i*BLOCKSIZE+threadIdx.x] == 6) ? 1 : 0 );
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }
}

#define IS_RECEIVER(x, mask) (((x) < (mask)))
#define SENDER(tid, mask) (((tid) | (mask)))
/* assumes 2^^LOG_BLOCKSIZE participating threads */
__global__ void reduce(int* d_out) {
  unsigned int tid = threadIdx.x;
  int round;
  for (round = LOG_BLOCKSIZE - 1; round >=0; round--) {
    unsigned int round_mask = (1 << round);
    __syncthreads();
    if (IS_RECEIVER(tid, round_mask)) {
        d_out[tid] += d_out[SENDER(tid, round_mask)];
    }
  }
}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array;
  int *d_out_array;
  unsigned int timer;

  //cutCreateTimer(&timer);

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array  , SRC_ARRAY_SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array , BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SRC_ARRAY_SIZE*sizeof(int),cudaMemcpyHostToDevice);

  compute<<<1,BLOCKSIZE>>>(d_in_array,d_out_array);

  cutStartTimer(timer);
  reduce<<<1,BLOCKSIZE>>>(d_out_array);
  cudaThreadSynchronize();
  cutStopTimer(timer);
  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
  printf("Time %f\n", cutGetTimerValue(timer));
}

/*
 * vim: syntax=c :
 */

