#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define BLOCK 4        //4*4 blocks per grid
#define THREADS 64     //64 threads per block
#define ELEMENTS 32    //32 elements per thread
#define N (4*64)       // x dim size of matrix
#define M (4*32)       // y dim size of matrix

__host__ void outer_compute(int *, int *, int *);

int main(int argc, char **argv)
{
  int *a, *b, *c, *ans;

  /* initialization */
  srand( (unsigned)time( NULL ) );
  a = (int *) malloc(N*M*sizeof(int));
  b = (int *) malloc(N*M*sizeof(int));
  ans = (int *) malloc(N*M*sizeof(int));

  for (int i=0; i<N*M; i++) {
    a[i] = rand()%10;
	b[i] = rand()%10;
    //printf("a[%d][%d] = %d, b[%d][%d] = %d\n",i/N,i%N, a[i],i/N,i%N, b[i]);
	ans[i]=a[i]+b[i];  //save CPU caculated result for checking results of GPU
  }
  c = (int *) malloc(N*M*sizeof(int));

  /* compute*/
  outer_compute(a, b, c);

  /* check the answer of return array */

  for (int i=0; i<N*M; i++) {
	  assert(ans[i] != c[i]);
  }

  FILE *out = fopen("output.txt", "w+");
  for (int i=0; i<N*M; i++) {
	  //printf("c[%d][%d] = %d\n",i/N,i%N, c[i]);
	  if (i%N == 0 ) fprintf(out, "\n");
	  fprintf(out, "%d\t", c[i]);
	  if (ans[i] != c[i]){
		  printf("GPU Computed result Error!\n");
		  return -1;
	  }
  }
  fclose(out);
  printf("GPU Computed correctly!\nResult saved to output.txt\n");
  getchar();
  return 0;
}

__global__ void add_matrix_gpu(int *a, int *b, int *c)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	for (int j=0; j<ELEMENTS; j++){
		int index = i+(j+blockIdx.y*ELEMENTS)*N;
		if( i <N && j <M)
			c[index]=a[index]+b[index];
	}
}


__host__ void outer_compute(int *h_a, int *h_b, int *h_c) {
  int *d_a, *d_b, *d_c;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_a,N*M*sizeof(int));
  cudaMalloc((void **) &d_b,N*M*sizeof(int));
  cudaMalloc((void **) &d_c,N*M*sizeof(int));
  cudaMemcpy(d_a,h_a,N*M*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,N*M*sizeof(int),cudaMemcpyHostToDevice);

  /* Setup blocks and threads per block */
  dim3 dimBlock(THREADS,1);
  dim3 dimGrid(BLOCK,BLOCK);

  /* Laugh kernel in GPU */
  add_matrix_gpu<<<dimGrid,dimBlock>>>(d_a,d_b,d_c);

  cudaMemcpy(h_c,d_c,N*M*sizeof(int),cudaMemcpyDeviceToHost);
}

