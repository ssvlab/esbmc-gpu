#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0;
  int ans=0;
  //srand( (unsigned)time( NULL ) );

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
	ans += (in_array[i]==6); //save GPU result for checking
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  /* GPU result now saved as out[0]*/
  sum = out_array[0];

  printf ("GPU: The number 6 appears %d times in array of  %d numbers.\n",sum,SIZE);
  printf ("CPU: The number 6 appears %d times in array of  %d numbers.\n",ans,SIZE);
  assert(sum == ans);
  getchar();
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__device__ int add(int a, int b) {
  return a+b;
}

__global__ void compute(int *d_in,int *d_out) {
  int i;

 /* compute number of appearances of 6 for subset of data in each thread! */
  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }

  /*reduce all results to d_out[0]*/
  for (i=1; i<BLOCKSIZE; i=i*2){
	  __syncthreads();
	  if (threadIdx.x % (2*i) == 0)
		  d_out[threadIdx.x] = add(d_out[threadIdx.x], d_out[threadIdx.x+i]);
  }

}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));
  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  /* run kernel on GPU */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

  /*copy oupt back to host memory*/
  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

