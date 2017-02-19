#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define THREADS 64 //# of threads executing in a block
#define ElEMENTS 32 //# of elements accessed by a thread
#define COLUMN (64*4) // # of elements in a row
#define ROW (32*4) // # of elements in a column

__host__ void add_matrix(int* h_a_m, int* h_b_m, int* h_c_m);

/* output the matrix into a file*/
void output_matrix(int* matrix, FILE* outputFile) {
   int i, j;

   for(i=0; i<ROW; i++) {
      for(j=0; j<COLUMN; j++) {
         fprintf(outputFile, "%d \t", matrix[i*COLUMN+j]);
      }
      fputc('\n', outputFile);
   }
}

/* Main function */
int main(int argc, char **argv) {
   int i;
   FILE* outputFile = fopen("output2.txt", "w");

   /* Initialization, CPU mem allocation */
   int * a_m = (int*)malloc(ROW*COLUMN*sizeof(int));
   int * b_m = (int*)malloc(ROW*COLUMN*sizeof(int));
   int * c_m = (int*)malloc(ROW*COLUMN*sizeof(int));

   for (i=0; i<ROW * COLUMN; i++) {
       a_m[i] = rand() % 10;
       b_m[i] = rand() % 10;
       c_m[i] = 0;
   }
   fputs("Matrix a is: \n", outputFile);
   output_matrix(a_m, outputFile);

   fputs("Matrix b is: \n", outputFile);
   output_matrix(b_m, outputFile);

   /* GPU kernel invocation */
   add_matrix(a_m, b_m, c_m);

   for (i=0; i<ROW * COLUMN; i++) {
	   assert(c_m[i] == 0);
   }

   fputs("Matrix c is: \n", outputFile);
   output_matrix(c_m, outputFile);
}

/* Compute the kernel */
__global__ void add_matrix_gpu(int *a, int *b, int *c)
{
   int k;
   int i = blockIdx.x*blockDim.x+threadIdx.x;

   for(k = 0; k < ElEMENTS; k++) {
     int j = blockIdx.y*ElEMENTS + k;
     int index =i+j*COLUMN;

     if( i<COLUMN && j<ROW)
       c[index]=a[index]+b[index];
   }
}

__host__ void add_matrix(int* h_a_m, int* h_b_m, int* h_c_m) {
    int *d_a_m, *d_b_m, *d_c_m;

    /* allocate memory for device copies, and copy input to device */
    int size = ROW*COLUMN*sizeof(int);
    cudaMalloc((void **) &d_a_m, size);
    cudaMalloc((void **) &d_b_m, size);
    cudaMalloc((void **) &d_c_m, size);

    cudaMemcpy(d_a_m, h_a_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_m, h_b_m, size, cudaMemcpyHostToDevice);

    /* ESBMC_execute the kernel */
    dim3 dimBlock(THREADS, 1);
    dim3 dimGrid(4, 4);
    add_matrix_gpu<<<dimGrid,dimBlock>>>(d_a_m,d_b_m,d_c_m);

    /* Copy results back to CPU */
    cudaMemcpy(h_c_m, d_c_m, size, cudaMemcpyDeviceToHost);
}

