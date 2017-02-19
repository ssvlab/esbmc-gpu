#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

FILE *fp;

__host__ void outer_compute(int *in_arr, int *out_arr);

int result;

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  fp = fopen("output.txt","w");

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));

  result = -1;

  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    fprintf(fp,"in_array[%d] = %d\n",i,in_array[i]);
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }

  out_array = (int *) malloc(SIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  fprintf(fp, "\nNumber 6 appears %d times  \n", out_array[0] );
  printf("\nNumber 6 appears %d times  \n", out_array[0] );

  assert(out_array[0] == 29);


  // printf ("The number 6 appears %d times in array of  %d numbers\n",sum,SIZE);
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out)
{
  d_out[blockIdx.x * BLOCKSIZE + threadIdx.x] = 0;
  d_out[blockIdx.x * BLOCKSIZE + threadIdx.x] = compare(d_in[blockIdx.x * BLOCKSIZE + threadIdx.x],6);
}

__global__ void reduce(int *d_out, int rIndex)
{
	int i = (blockIdx.x * BLOCKSIZE) + threadIdx.x;
	int jump = rIndex / 2;

	if( (i == 0) || ( (i/rIndex) && (i%rIndex == 0) )  )
	{
		int j = (blockIdx.x * BLOCKSIZE + threadIdx.x) + jump;
		d_out[i] += d_out[j];
	}
}



__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,SIZE*sizeof(int));

  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

  dim3 bSize(BLOCKSIZE);
  dim3 gSize(SIZE/BLOCKSIZE);


  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<gSize,bSize,0>>>(d_in_array,d_out_array);


  for(int i = 2;i <= SIZE; i *= 2)
  	  reduce<<<gSize,bSize,0>>>(d_out_array,i);

  cudaMemcpy(h_out_array,d_out_array,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
}

