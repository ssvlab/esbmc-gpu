#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

typedef int myint;
#define NUM_THREADS 64
#define NUM_ELEMENTS_PER_THREAD 32
#define NUM_BLOCKS 16

/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ static void add_matrix_gpu(myint *a, myint *b, myint *c, int N)
{
	int i = blockIdx.x ;
	int j = blockIdx.y * gridDim.y;
	int index = (i + j) * (NUM_THREADS * NUM_ELEMENTS_PER_THREAD);
	index += threadIdx.x * NUM_ELEMENTS_PER_THREAD;
	for(int k=0; k < NUM_ELEMENTS_PER_THREAD; k++)
	{
		c[index + k] = a[index + k] + b[index + k];
	}
}



/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{


	myint* host_a;
	myint* host_b;
	myint* host_c;
	int N = NUM_THREADS * NUM_ELEMENTS_PER_THREAD * NUM_BLOCKS;  //64 threads * 32 elements(worked on per thread) * 16 blocks(of threads)

	host_a = (myint*)malloc(sizeof(myint) * N);
	host_b = (myint*)malloc(sizeof(myint) * N);
	host_c = (myint*)malloc(sizeof(myint) * N);

	//initialize a, b to teh same
	for(int i=0; i < N; i++)
	{
		host_a[i] = host_b[i] = rand() % 10;
		host_c[i] = 0;
	}

	//allocate mem on device
	myint* d_a;
	myint* d_b;
	myint* d_c;
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_a, sizeof(myint)*N) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_b, sizeof(myint)*N) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_c, sizeof(myint)*N) );

	//copy data from host to device
	CUDA_SAFE_CALL( cudaMemcpy(d_a, host_a, sizeof(myint) * N, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_b, host_b, sizeof(myint) * N, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_c, host_c, sizeof(myint) * N, cudaMemcpyHostToDevice));

	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	//dim3 dimBlock(4,4);
	dim3 dimGrid(4,4);
	add_matrix_gpu<<<dimGrid,64>>>(d_a, d_b, d_c, N);

	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
	CUT_SAFE_CALL( cutDeleteTimer( timer));

	//copy results from device
	CUDA_SAFE_CALL( cudaMemcpy(host_c, d_c, sizeof(myint) * N, cudaMemcpyDeviceToHost));

	//check the results
	bool iscorrect = true;
	int numw = 0;
	for(int i=0; i < N; i++)
	{	assert(host_c[i] == 0);
		if(host_c[i] != host_a[i] + host_b[i])
		{
			iscorrect = false;
			numw++;
		}
	}

	if(iscorrect)
	{
		printf("IS CORRECT");
	}else
	{
		printf("IS NOOOOT CORRECT  there are %d wrong",numw);
	}
	CUDA_SAFE_CALL( cudaFree(d_a));
	CUDA_SAFE_CALL( cudaFree(d_b));
	CUDA_SAFE_CALL( cudaFree(d_c));
	free(host_a);
	free(host_b);
	free(host_c);
	CUT_EXIT(argc, argv);

	getchar();
	return 0;
}

