#include <call_kernel.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32
#define SEARCH_NUMBER 5

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0;

  //srand(time(NULL));

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }


  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of the number */
  //calling the host function
  outer_compute(in_array, out_array);

  //for (int i=0; i<BLOCKSIZE; i++) {
  //  sum+=out_array[i];
  //}

  printf ("The number %d appears %d times in array of %d numbers\n",(int)SEARCH_NUMBER,out_array[0],SIZE);

  if(SEARCH_NUMBER < 10)
  	  assert(out_array[0] != 0);

  getchar();
  return 0;
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out) {
  int i;
  int pos, incr;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],SEARCH_NUMBER);
  }


  //sync the threads before the reduction
  __syncthreads();

  //height of the binary tree given by log2(#leaves)
  for(i=0; i<(int)log2((float)BLOCKSIZE); i++)
  {
    //math
	pos = (int)pow(2.0,i+1);
	incr = (int)pow(2.0,i);

	//if this is the thread which needs to be updated
	if(threadIdx.x%pos == 0)
	{
		d_out[threadIdx.x] += d_out[threadIdx.x + incr];
	}

	//sync the threads before starting the next iteration
	__syncthreads();
  }


}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

