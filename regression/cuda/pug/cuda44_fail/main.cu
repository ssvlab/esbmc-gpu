#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
//#include <cutil.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr, int *out_value);

int main(int argc, char **argv)
{
  int *in_array, *out_array, *out_total;
  int sum=0, sumCPU=0;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    if (in_array[i]==6)
    	sumCPU++;
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));
  out_total = (int *) malloc(sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array, out_total);

  sum=out_total[0];
  printf ("\nThe number 6 appears %d times in array of  %d numbers (should be %d)\n",sum,SIZE,sumCPU);

  assert(sum!=sumCPU);
  //CUT_EXIT(argc, argv);
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out, int *d_sum) {
  int i;
  *d_sum=0;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);

      //printf ("%d;",d_out[threadIdx.x]);
  }

  __syncthreads();


  for(i=0; i<BLOCKSIZE; i++){
	  d_sum[0] += d_out[i];
  }
}

__host__ void outer_compute(int *h_in_array, int *h_out_array, int *h_out_value) {
  int *d_in_array, *d_out_array, *d_out_value;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMalloc((void **) &d_out_value,sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array,d_out_value);

  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_value,d_out_value,sizeof(int),cudaMemcpyDeviceToHost);
}



