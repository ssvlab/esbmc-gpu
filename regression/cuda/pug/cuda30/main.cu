#include <call_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define SIZEX 256
#define SIZEY 128
#define BLOCKSIZE 64
#define GRIDSIZE 4
//#define SIZE 16
//#define BLOCKSIZE 4

__host__ void outer_compute(int *in_arr, int *in_arg, int *out_arr);

int main(int argc, char **argv)
{
  int *mat_a, *mat_b;
  int *sum;

	FILE *fp;
	fp = fopen("orcwarrior.txt", "w");

  /* initialization */
  mat_a = (int *) malloc(SIZEX*SIZEY*sizeof(int));
  mat_b = (int *) malloc(SIZEX*SIZEY*sizeof(int));
  sum = (int *) malloc(SIZEX*SIZEY*sizeof(int));
  for (int i=0; i<SIZEX*SIZEY; i++) {
    mat_a[i] = rand()%10;
	mat_b[i] = rand()%10;
	sum[i] = 0;
  }

  /* compute number of appearances of 6 */
  outer_compute(mat_a, mat_b, sum);

//  printf ("The number 6 appears %d times in array of  %d %d numbers\n",sum[537],mat_a[537], mat_b[537]);
	for (int i=0; i<SIZEX*SIZEY; i++) {
		if (i % SIZEX == 0) printf("\n");
		fprintf(fp, "%d + %d = %d, ", mat_a[i], mat_b[i], sum[i]);
	}
	fprintf(fp, "\n");
	for (int i=0; i<SIZEX*SIZEY; i++) {
		if (i % SIZEX == 0) printf("\n");
		fprintf(fp, "%d, ", sum[i]);
	}

	fclose(fp);

	for (int i=0; i<SIZEX*SIZEY; i++) {
	   assert(sum[i] == mat_a[i] + mat_b[i]);


	  }

	CUT_EXIT(argc, argv);

	return 0;
}

__global__ void compute(int *d_in_a, int *d_in_b, int *d_out) {
  int i, index;

	index = blockIdx.x*SIZEX/GRIDSIZE + blockIdx.y*SIZEX*SIZEY/GRIDSIZE+ threadIdx.x;

	for (i = 0; i < SIZEY/GRIDSIZE; i++) {
		d_out[index] = d_in_a[index] + d_in_b[index];
		index += SIZEX;
	}
}

__host__ void outer_compute(int *h_in_mat_a, int *h_in_mat_b, int *h_out_array) {
  int *d_in_mat_a, *d_in_mat_b, *d_out_array;

	dim3 blocks(BLOCKSIZE,1);
	dim3 gridsize(GRIDSIZE,GRIDSIZE);

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_mat_a,SIZEX*SIZEY*sizeof(int));
  cudaMalloc((void **) &d_in_mat_b,SIZEX*SIZEY*sizeof(int));
  cudaMalloc((void **) &d_out_array,SIZEX*SIZEY*sizeof(int));
  cudaMemcpy(d_in_mat_a,h_in_mat_a,SIZEX*SIZEY*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_mat_b,h_in_mat_b,SIZEX*SIZEY*sizeof(int),cudaMemcpyHostToDevice);


  compute<<<gridsize,blocks,0>>>(d_in_mat_a,d_in_mat_b,d_out_array);

	// only copy the first value of the output array here since that's all we need.
  cudaMemcpy(h_out_array,d_out_array,SIZEX*SIZEY*sizeof(int),cudaMemcpyDeviceToHost);
}

