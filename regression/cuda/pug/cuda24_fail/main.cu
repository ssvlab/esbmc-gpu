#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define SIZE 16
#define ELEMENTS 16*64*32
#define BLOCKSIZE 64
#define COLUMNS 128
#define ROWS 256
#define ELEMENTS_PER_THREAD 32
#define MAT_COL_PER_BLOCK 8


/***************************************************************/
__global__ void add_matrix_gpu(int *a, int *b, int *c, int N)
{
   int i;
   int j;
   int index;
   int k = threadIdx.x * ELEMENTS_PER_THREAD;


   // There are 4x4 blocks. 8 columns of 32 elemens in each
   // block form one column (256 element) of a matrix. Therefore,
   // each block contains 8 columns of a matrix.

   j=(blockIdx.y*4 + blockIdx.x) * MAT_COL_PER_BLOCK;

   for( i=0; i<ELEMENTS_PER_THREAD; i++ ) {
     index =i+k+j*N;
     c[index]=a[index]+b[index];
   }
}

/***************************************************************/
int main() {

  //unsigned int timer = 0;
  dim3 dimGrid(4,4);
  int  mem_size=ELEMENTS * sizeof(int);
  int  *h_a;
  int  *h_b;
  int  *h_c;
  int  *d_a;
  int  *d_b;
  int  *d_c;


  /* allocate space for host matrices */
  h_a = (int *) malloc(mem_size);
  h_b = (int *) malloc(mem_size);
  h_c = (int *) malloc(mem_size);

  /* initialization */
  for (int i=0; i<ELEMENTS; i++) {
    h_a[i] = rand()%10;
    h_b[i] = rand()%10;
    h_c[i] = 0;
  }

  /* allocate space for device matrices */
  cudaMalloc((void**) &d_a, mem_size);
  cudaMalloc((void**) &d_b, mem_size);
  cudaMalloc((void**) &d_c, mem_size);

  /* copy host memory to device */
  cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);

  /* create and start timer */
  //cutCreateTimer(&timer);
  //cutStartTimer(timer);

  /* ESBMC_execute the kernel */
  add_matrix_gpu<<<dimGrid,BLOCKSIZE,0>>>(d_a, d_b, d_c, ROWS);

  /* stop and destroy timer */
  //cutStopTimer(timer);
  //printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
  //cutDeleteTimer(timer);

  /* copy result from device to host */
  cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost);

  /* print out the result */
  for (int i=0; i<ELEMENTS; i++) {
    printf("a[%d] + b[%d] = %d + %d = c[%d] = %d\n",i,i,h_a[i], h_b[i], i, h_c[i]);
    assert(h_c[i] == 0);
  }

  /* clean up */
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  getchar();
}

