#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);
__host__ void outer_calcSum(int *in_arr, int *out_arr, int in_arr_size);

/***************************************************************/
int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int *tmp_array;
  int threads=0;
  int sum=0;
  int times=0;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = random()%10;
    if (in_array[i] == 6)
    	times += 1;
    /* printf("in_array[%d] = %d\n",i,in_array[i]); */
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

#ifdef ORIGINAL
  for (int i=0; i<BLOCKSIZE; i++) {
    sum+=out_array[i];
  }
#else

  /* allocate memory for the final sum */
  tmp_array = (int *) malloc((BLOCKSIZE/2)*sizeof(int));

  threads = BLOCKSIZE;
  while ( threads > 2) {

    /* compute the sume of appearances of 6 */
    outer_calcSum(out_array, tmp_array, threads);

    threads = threads / 2;
    memcpy(out_array, tmp_array, threads*sizeof(int));
  }
  sum = out_array[0] + out_array[1];

  free(tmp_array);
#endif

  printf ("The number 6 appears %d times in an array of  %d numbers\n",sum,SIZE);
  assert(sum != times);

  /* clean up */
  free(in_array);
  free(out_array);

}

/***************************************************************/
__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

/***************************************************************/
__global__ void compute(int *d_in,int *d_out) {
  int i;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }
}

/***************************************************************/
__global__ void calcSum(int *d_in,int *d_out) {
  int i;

  i = threadIdx.x * 2;
  d_out[threadIdx.x] = d_in[i] + d_in [i+1];
}

/***************************************************************/
__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

/***************************************************************/
__host__ void outer_calcSum(int *h_in_array, int *h_out_array, int in_arr_size) {
  int half=in_arr_size / 2;
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array, in_arr_size*sizeof(int));
  cudaMalloc((void **) &d_out_array, half*sizeof(int));
  cudaMemcpy(d_in_array, h_in_array, in_arr_size*sizeof(int), cudaMemcpyHostToDevice);

  /* calculate the sum of appearances of 6 for subset of data in each thread! */
  calcSum<<<1,half,(in_arr_size+half)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array, d_out_array, half*sizeof(int), cudaMemcpyDeviceToHost);
}

