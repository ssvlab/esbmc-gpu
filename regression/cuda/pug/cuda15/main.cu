#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define GRIDSIZE_X 4
#define GRIDSIZE_Y 4
#define BLOCKSIZE 64
#define BLOCKCOLUMNSIZE 32

#define SIZE_X (GRIDSIZE_X * BLOCKSIZE)
#define SIZE_Y (GRIDSIZE_Y * BLOCKCOLUMNSIZE)
#define SIZE (SIZE_X * SIZE_Y)
#define GRIDCOLUMNSIZE (SIZE_Y * BLOCKSIZE)

#define DEBUG_MODE true


__host__ void outer_add_matrix_gpu(int *in_mat_a, int *in_mat_b, int *out_mat_c);


void printMatrix(char id, int *A, FILE *fp)
{
  fprintf(fp,"--------------------------------------------------------------------------------\n");
  fprintf(fp,"----------------------------------- Matrix %c -----------------------------------\n", id);
  fprintf(fp,"--------------------------------------------------------------------------------\n");
  for(int i=0; i<SIZE_X; i++) {
    for(int j=0; j<SIZE_Y; j++)
      fprintf(fp,"%d\t", A[i*SIZE_Y+j]);
    fprintf(fp,"\n");
  }
  fprintf(fp,"================================================================================\n");
  fprintf(fp,"\n\n");
}


int main(int argc, char **argv)
{
  /* Allow filename as first argument - default to stdout if argument is missing or invalid */
  FILE *file;
  if (argc > 1) {
    file = fopen(argv[1], "w");
    if (file == NULL) {
      fprintf(stderr, "Failed to open file \"%s\" for writing.  Writing to stdout instead.\n", argv[1]);
      file = stdout;
    } else {
      fprintf(stdout, "Writing to file: \"%s\"\n", argv[1]);
    }
  }
  else {
    file = fopen("output.txt","w");
  }

  int *in_matrix_a, *in_matrix_b, *out_matrix_c;

  /* seed the random number generator */
  srand(1); // ISO-C default seed

  /* initialization */
  in_matrix_a = (int *) malloc(SIZE*sizeof(int));
  in_matrix_b = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_matrix_a[i] = rand()%10;
    in_matrix_b[i] = rand()%10;
  }

  out_matrix_c = (int *) malloc(SIZE*sizeof(int));

  /* compute matrix addition */
  outer_add_matrix_gpu(in_matrix_a, in_matrix_b, out_matrix_c);

  /* print matrices */
  printMatrix('A', in_matrix_a, file);
  printMatrix('B', in_matrix_b, file);
  printMatrix('C', out_matrix_c, file);

   /* close the output file */
  if (file != stdout) {
    fclose(file);
  }

  for (int i=0; i<SIZE; i++)
  	  assert(out_matrix_c[i] == in_matrix_a[i] + in_matrix_b[i]);

  if (DEBUG_MODE) {
    /* Verify results on the CPU */
    bool correct = true;
    int checkValue;
    for (int i=0; i<SIZE; i++) {
      checkValue = in_matrix_a[i] + in_matrix_b[i];
      if (out_matrix_c[i] != checkValue) {
        correct = false;
        fprintf(stderr, "FAILED: GPU addition incorrect at index [%d]. Should be %d, but was %d.\n", i, checkValue, out_matrix_c[i]);
      }
    }
    if (correct) {
      fprintf(stdout, "PASSED: GPU addition matched CPU addition.\n");
	  getchar();
    }
  }

}


__global__ void add_matrix_gpu(int *d_in_matrix_a, int *d_in_matrix_b, int *d_out_matrix_c) {
  int i;

  /* Compute the starting and stopping indices for this thread */
  int startIndex = blockIdx.x  * GRIDCOLUMNSIZE
                 + threadIdx.x * SIZE_Y
                 + blockIdx.y  * BLOCKCOLUMNSIZE;

  int stopIndex = startIndex + BLOCKCOLUMNSIZE;

  /* Each thread runs through BLOCKCOLUMNSIZE elements in order (block data distribution) */
  for (i=startIndex; i<stopIndex; i++) {
    d_out_matrix_c[i] = d_in_matrix_a[i] + d_in_matrix_b[i];
  }

}


__host__ void outer_add_matrix_gpu(int *h_in_matrix_a, int *h_in_matrix_b, int *h_out_matrix_c) {
  int *d_in_matrix_a, *d_in_matrix_b, *d_out_matrix_c;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_matrix_a,  SIZE*sizeof(int));
  cudaMalloc((void **) &d_in_matrix_b,  SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_matrix_c, SIZE*sizeof(int));
  cudaMemcpy(d_in_matrix_a, h_in_matrix_a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_matrix_b, h_in_matrix_b, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  /* compute matrix addition */
  dim3 dimGrid(GRIDSIZE_X, GRIDSIZE_Y);
  //add_matrix_gpu<<<dimGrid,BLOCKSIZE,0>>>(d_in_matrix_a, d_in_matrix_b, d_out_matrix_c);
	ESBMC_verify_kernel_with_three_args(add_matrix_gpu, dimGrid, BLOCKSIZE, d_in_matrix_a, d_in_matrix_b, d_out_matrix_c);

  /* copy computed matrix back to host */
  cudaMemcpy(h_out_matrix_c, d_out_matrix_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
}

