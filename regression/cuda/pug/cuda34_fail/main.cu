#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//#include <cutil.h>
#include <assert.h>


__host__ void outer_compute(int * a , int *b, int *c, int N, int M);

__device__ int gpuadd(int a , int b)
{
	return a+b;
}

__global__ void add_matrix_gpu(int *a, int *b, int *c, int N, int M )
{
	//int i = blockIdx.x*blockDim.x+threadIdx.x;
	//int j=blockIdx.y*blockDim.y+threadIdx.y;
	//int index =i+j*N;
	//if( i <N && j <M)
	//	c[index]=a[index]+b[index];
	int k;
	int index;
	for(k= 0; k < 32; k++)
	{
		index = N*(blockIdx.y*32+k) + blockIdx.x * 64 + threadIdx.x;
		c[index] = gpuadd(a[index],b[index]);
	}
}


int main() {

	int *a, *b, *c, *ans;
	int M = 128;
	int N = 256;

  srand( (unsigned)time( NULL ) );



  /* initialization */
  a = (int *) malloc(M*N*sizeof(int));
  b = (int *) malloc(M*N*sizeof(int));
  for (int i=0; i<N*M; i++) {
    a[i] = rand()%10;
	b[i] = rand()%10;
  }
  c = (int *) malloc(M*N*sizeof(int));
  ans = (int *) malloc(M*N*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(a,b,c,N,M);

  int i;
  for(i = 0; i< N*M; i++)
  {

	ans[i] = a[i] + b[i];
  }

  for (i=0;i<N*M;i++)
  {
	  if(c[i] != ans[i])
	  {
		  printf("NO!\n");
		  break;

	  }

	  assert(c[i] != ans[i]);
  }

  int k;
  FILE *output = fopen("outputfile.txt","w+");

  fprintf(output,"Matrix a:\n");
  for(i = 0; i< N*M; i++)
  {

	     fprintf(output,"%d ",a[i]);


  }

  fprintf(output,"\n\n\n\n");

  fprintf(output,"Matrix b:\n");

  for(i = 0; i< N*M; i++)
  {

	     fprintf(output,"%d ",b[i]);


  }

  fprintf(output,"\n\n\n\n");

  fprintf(output,"Addition of two Matrices :\n");

  for(i = 0; i< N*M; i++)
  {

	     fprintf(output,"%d ",c[i]);


  }



  fclose(output);
  //for(i = 0; i< N*M; i++)
  //{
	 //printf("%d ",a[i]);

  //}

  //printf("\n\n");

  //for(i = 0; i< N*M; i++)
  //{
	 //printf("%d ",b[i]);

  //}

  //printf("\n\n");

  //for(i = 0; i< N*M; i++)
  //{
	 //printf("%d ",c[i]);

  //}

  //printf("\n\n");

  //for(i = 0; i< N*M; i++)
  //{
	 //printf("%d ",ans[i]);

  //}

	return 0;
}

__host__ void outer_compute(int *h_a, int *h_b, int *h_c, int N, int M) {
  int *d_a, *d_b, *d_c;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_a,N*M*sizeof(int));
  cudaMalloc((void **) &d_b,N*M*sizeof(int));
  cudaMalloc((void **) &d_c,N*M*sizeof(int));
  cudaMemcpy(d_a,h_a,N*M*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,N*M*sizeof(int),cudaMemcpyHostToDevice);


  dim3 dimBlock(64,1);
  dim3 dimGrid(4,4);
  add_matrix_gpu<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,N,M);

  cudaMemcpy(h_c,d_c,N*M*sizeof(int),cudaMemcpyDeviceToHost);
}

