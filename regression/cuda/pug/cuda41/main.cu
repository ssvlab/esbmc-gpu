#include <call_kernel.h>
#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0, sumCPU=0;
  char forEnd;
  FILE * outfile;

  /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    if (in_array[i] == 6)
    	sumCPU++;
    printf("in_array[%d] = %d\n",i,in_array[i]);
  }
  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  sum = out_array[0];

  outfile = fopen ("./Assignment1_Problem1_output.txt", "w");
  printf ("The number 6 appears %d times in array of  %d numbers\n",sum,SIZE);
  fprintf (outfile, "The number 6 appears %d times in array of  %d numbers\n",sum,SIZE);
  fclose(outfile);
  assert(sumCPU == sum);
  printf("Press and key for end");
  scanf("%c", &forEnd);

}

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}

__global__ void compute(int *d_in,int *d_out) {
  int i;

  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
  }
}

__global__ void converge(int *d_out, int scope, int gap) {
	int initPoint = threadIdx.x * scope;
	int endPoint = initPoint + scope;
	int i;

	for (i = initPoint+gap ; i < endPoint ; i=i+gap) {
		d_out[initPoint] += d_out[i];
	}
}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;
  // ==== variables for converge ====
  int step = 2;				// the increase times of scope
  int scope = 2;			// the size of range of the sum need to calculate
  int gap = 1;				// the gap between datas
  int n_section;			// number of threads will be created
  // ==== end of variables for converge ====

  /* allocate memory for device copies, and copy input to device */
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_in_array,SIZE*sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice));

  /* compute number of appearances of 6 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  do {
	n_section = BLOCKSIZE%scope == 0 ? BLOCKSIZE/scope : BLOCKSIZE/scope + 1;			// decide the number of threads be created
	converge<<<1, n_section, (BLOCKSIZE)*sizeof(int)>>>(d_out_array, scope, gap);		// create the threadZ
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	gap = scope;					// the gap between the datas now be scope
	scope = scope * step;
  } while (scope < BLOCKSIZE);		// if the scope >= BLOCKSIZE -> let's do the last time
  converge<<<1, 1, (BLOCKSIZE)*sizeof(int)>>>(d_out_array, scope, gap);				// the last time -> only one thread be created
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost));
}

