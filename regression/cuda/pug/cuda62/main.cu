#include <call_kernel.h>
#include <stdio.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0, count=0;
//  srand( (unsigned)time( NULL ) );



  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);
    if (in_array[i] == 6)
    	count++;
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  for (int i=0; i<BLOCKSIZE; i++) {
    sum+=out_array[i];
  }

  FILE *output = fopen("output_1.txt", "w+");
  fprintf(output, "Input array:\n");
  int i;
  for(i=0;i<SIZE;i++)
  {
	fprintf(output,"%d ", in_array[i]);
  }

  fprintf(output, "\n\n\n");

  fprintf(output, "Output:\n");

	fprintf(output, "%d ", out_array[0]);


  fclose(output);

  printf ("The number 6 appears %d times in array of  %d numbers\n",out_array[0],SIZE);

  assert(out_array[0] == count);
}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__device__ int gpuadd(int a, int b){
	return a + b;
}

__global__ void compute(int *d_in,int *d_out) {
	int i;
	int j;

	d_out[threadIdx.x] = 0;
	for (i=0; i<SIZE/BLOCKSIZE; i++) {
		d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
	}


	__syncthreads();
	for (j = 1; j < BLOCKSIZE; j *= 2)
	{

		if(threadIdx.x % (2 * j) == 0)
			d_out[threadIdx.x] = gpuadd(d_out[threadIdx.x], d_out[threadIdx.x + j]);
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

