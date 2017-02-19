#include <call_kernel.h>
#include <stdio.h>
#include <time.h>
#include <cutil.h>
#include <assert.h>

#define M 128 //number of rows
#define N 256 //number of cols
#define MATRIX_SIZE 32768
#define NUMBER_OF_BLOCKS 4
#define NUMBER_OF_THREADS_X 64
#define ELEMENTS_PER_THREAD 32

__global__ void add_matrix_gpu(float *a, float *b, float *c);

int main(int argc, char **argv)
{
  float *a, *b, *c;
  FILE *op_file;
  int count;
  dim3 dimGrid(NUMBER_OF_BLOCKS, NUMBER_OF_BLOCKS);
  dim3 dimBlock(NUMBER_OF_THREADS_X, 1);
  unsigned int timer;
  float time_taken;

  op_file = fopen("output.txt","w");
  srand(time(NULL));

  /* Matrix A initialization */
  a = (float *) malloc(MATRIX_SIZE*sizeof(float));
  count = 0;
  fprintf(op_file, "\n---------------------------------------------------------------------------------\n");
  fprintf(op_file, "Matrix A\n");
  fprintf(op_file, "---------------------------------------------------------------------------------\n\n");
  for (int i=0; i<MATRIX_SIZE; i++) {
    a[i] = rand()%10;
    fprintf(op_file,"a[%d] = %5f\t",i,a[i]);
    count++;
    if(count == N)
    {
		count = 0;
		fprintf(op_file,"\n");
	}
  }


  // Matrix B initialization
  b = (float *) malloc(MATRIX_SIZE*sizeof(float));
  count = 0;
  fprintf(op_file, "\n---------------------------------------------------------------------------------\n");
  fprintf(op_file, "Matrix B\n");
  fprintf(op_file, "---------------------------------------------------------------------------------\n\n");
  for (int i=0; i<MATRIX_SIZE; i++) {
    b[i] = rand()%10;
    fprintf(op_file,"b[%d] = %5f\t",i,b[i]);
    count++;
    if(count == N)
    {
		count = 0;
		fprintf(op_file,"\n");
	}
  }

  //Starting the timer
  //cutCreateTimer(&timer);
  //cutStartTimer(timer);

  //allocating memory for Matrix C
  c = (float *) malloc(MATRIX_SIZE*sizeof(float));

  for (int i=0; i<MATRIX_SIZE; i++) {
      c[i] = 0;
  }

  //allocating the memory on the device
  float *dev_a, *dev_b, *dev_c;

  cudaMalloc((void **) &dev_a,MATRIX_SIZE*sizeof(float));
  cudaMalloc((void **) &dev_b,MATRIX_SIZE*sizeof(float));
  cudaMalloc((void **) &dev_c,MATRIX_SIZE*sizeof(float));

  cudaMemcpy(dev_a,a,MATRIX_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b,b,MATRIX_SIZE*sizeof(float),cudaMemcpyHostToDevice);

  //calling the device function
  add_matrix_gpu<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);

  cudaMemcpy(c,dev_c,MATRIX_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

  //Stopping the timer
  //cudaThreadSynchronize();
  //cutStopTimer(timer);
  //time_taken = cutGetTimerValue(timer);
  //printf("Time taken = %f\n",time_taken);

  for (int i=0; i<MATRIX_SIZE; i++) {
     assert(c[i] == a[i] + b[i]);
  }

  //Printing the output
  count = 0;
  fprintf(op_file, "\n---------------------------------------------------------------------------------\n");
  fprintf(op_file, "Matrix C\n");
  fprintf(op_file, "---------------------------------------------------------------------------------\n\n");
  for (int i=0; i<MATRIX_SIZE; i++) {
    if(c[i] >= 0)
    {
		fprintf(op_file,"c[%d] = %5f\t",i,c[i]);
		count++;
		if(count == N)
		{
			count = 0;
			fprintf(op_file,"\n");
		}
    }
  }

  fclose(op_file);
  printf("output.txt has been written successfully\n");
  getchar();
  return 0;
}



//Device function implementation
__global__ void add_matrix_gpu(float *a, float *b, float *c) {

  int i,j, index;

  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;

  index = i+j*(NUMBER_OF_THREADS_X * ELEMENTS_PER_THREAD * NUMBER_OF_BLOCKS);

  if(i<N && j<NUMBER_OF_BLOCKS)
  {
    //Each thread handles the addition of 32 elements in the Matrix
	for(int k=1; k<=ELEMENTS_PER_THREAD; k++)
	{
		if(index < MATRIX_SIZE)
		{
			c[index] = a[index] + b[index];
			index = index + N;
		}
	}
  }

}

