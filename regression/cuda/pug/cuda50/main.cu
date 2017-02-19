#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define INIT_SIZE 256
#define INIT_BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum;
  /* initialization */
  in_array = (int *) malloc(INIT_SIZE*sizeof(int));
  for (int i=0; i<INIT_SIZE; i++) {
    in_array[i] = rand()%10;
	printf("in_array[%d] = %d\n",i,in_array[i]);
	if (in_array[i] == 6)
		sum +=1;
  }
  out_array = (int *) malloc(INIT_BLOCKSIZE*sizeof(int));

  outer_compute(in_array, out_array);
  printf ("Sum: CUDA: %d\n",out_array[0] );
  assert(out_array[0] == sum);
  getchar();
}

__global__ void compute(int *d_in,int *d_out) {

	int blocksize = INIT_BLOCKSIZE;
	int size = INIT_SIZE;
	int coalesce_num = size/blocksize;
	d_out[threadIdx.x] = 0;
	// 256->32
	for (int j=0; j < coalesce_num; j++)
		d_out[threadIdx.x] += (d_in[j * blocksize + threadIdx.x] == 6)?1:0;

	//32->16, 16->8, 8->4, 4->2, 2->1
	for (int i=0; i<5; i++){
		__syncthreads();
		blocksize = blocksize/2;
		__syncthreads();

		if (threadIdx.x < blocksize){
			d_out[threadIdx.x] += d_out[ blocksize + threadIdx.x];
		}
	}
}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,INIT_SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,INIT_BLOCKSIZE*sizeof(int));
  cudaMemset(d_out_array, 0,INIT_BLOCKSIZE * sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,INIT_SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,INIT_BLOCKSIZE,(INIT_SIZE+INIT_BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array,d_out_array,INIT_BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

