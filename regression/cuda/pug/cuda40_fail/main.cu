#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define BLOCKSIZE 256

__host__ void outer_compute(int *in_arr, int *out_value);

int main(int argc, char **argv)
{
  int *in_array, *out_total;
  int sum=0;
  int sum_cpu=0;

  /* initialization */
  in_array = (int *) malloc(BLOCKSIZE*sizeof(int));
  out_total = (int *) malloc(BLOCKSIZE*sizeof(int));
  for (int i=0; i<BLOCKSIZE; i++) {
    in_array[i] = rand()%10;
    *out_total = 0;
    printf("in_array[%d] = %d\n",i,in_array[i]);
    if (in_array[i] == 6)
    	sum_cpu++;
  }

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_total);

  sum=out_total[0];
  printf ("\nThe number 6 appears %d times in array of %d numbers (should be %d)\n",sum,BLOCKSIZE, sum_cpu);
  assert(sum != sum_cpu);
  getchar();
}

__device__ int compare(int a, int b) {
  if (a == b)
	return 1;
  else
	return 0;
}

__global__ void compute(int *d_in, int *d_sum) {

	d_sum[threadIdx.x] += compare(d_in[threadIdx.x],6);
	 __syncthreads();
	 for (int i=0; i<BLOCKSIZE; i++){
		 d_sum[0] += d_sum[i];
	 }

 __syncthreads();

}



__host__ void outer_compute(int *h_in_array, int *h_out_value) {
  int *d_in_array, *d_out_value;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,BLOCKSIZE*sizeof(int));
  cudaMalloc((void **) &d_out_value,BLOCKSIZE*sizeof(int));

  cudaMemcpy(d_in_array,h_in_array,BLOCKSIZE*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_value,h_out_value,BLOCKSIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE>>>(d_in_array,d_out_value);

  cudaMemcpy(h_out_value,d_out_value,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

