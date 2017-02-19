#include <call_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define SIZE 256
#define BLOCKSIZE 32
#define COMPARE_VAL 6 // what number should be counted! used in both GPU and CPU calculations

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}
#endif

// Count the number of 6's!
__global__ static void count6s(int *inArray, int *threadArray, int *count) {
	threadArray[threadIdx.x] = 0;

	for(int i = 0; i < (SIZE/BLOCKSIZE); i++) {
		int val = inArray[i*BLOCKSIZE+threadIdx.x];
		if(val == COMPARE_VAL)
			threadArray[threadIdx.x]++;
	}

	__syncthreads();

	int k = BLOCKSIZE/2;
	int i = 1;
	for(; k >= 1; k /= 2) {
		int kRev = BLOCKSIZE/k;
		if((threadIdx.x % kRev) == 0){
			threadArray[threadIdx.x] += threadArray[threadIdx.x+i];
		}
		i *= 2;
		__syncthreads();
	}

	if(threadIdx.x == 0)
		*count = threadArray[0];
}

/************************************************************************/
/* Count the number of 6's!                                                          */
/************************************************************************/
int main(int argc, char* argv[])
{

	if(!InitCUDA()) {
		return 0;
	}

	int *dataArray = 0;
	dataArray = (int *)malloc(SIZE*sizeof(int));
	printf("Array contents...\n(");
	for(int i = 0; i < (SIZE-1); i++) {
		dataArray[i] = rand()%10;
		printf( "%d, ", dataArray[i] );
	}
	dataArray[SIZE-1] = rand()%10;
	printf( "%d", dataArray[SIZE-1] );
	printf(")\n");

	int *inArray = 0;
	CUDA_SAFE_CALL( cudaMalloc((void**) &inArray, sizeof(int)*SIZE) );
	CUDA_SAFE_CALL( cudaMemcpy(inArray, dataArray, sizeof(int)*SIZE, cudaMemcpyHostToDevice) );
	int *threadArray = 0;
	CUDA_SAFE_CALL( cudaMalloc((void**) &threadArray, sizeof(int)*BLOCKSIZE) );
	int *count = 0;
	CUDA_SAFE_CALL( cudaMalloc((void**) &count, sizeof(int)) );


	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer(&timer) );
	CUT_SAFE_CALL( cutStartTimer(timer) );

	// call the count function
	count6s<<<1, BLOCKSIZE, (SIZE+BLOCKSIZE+1)*sizeof(int)>>>(inArray, threadArray, count);
	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutStopTimer(timer) );
	printf( "Processing time (count6): %f (ms)\n", cutGetTimerValue(timer) );
	CUT_SAFE_CALL( cutDeleteTimer( timer ) );

	int output = 0;
	CUDA_SAFE_CALL( cudaMemcpy(&output, count, sizeof(int), cudaMemcpyDeviceToHost) );

	printf( "Number of times %d appears in the array(GPU calculation): %d \n", COMPARE_VAL, output);

	CUDA_SAFE_CALL( cudaFree(inArray) );
	CUDA_SAFE_CALL( cudaFree(threadArray) );
	CUDA_SAFE_CALL( cudaFree(count) );

	int countCPU = 0;
	for(int i = 0; i < SIZE; i++) {
		if(dataArray[i] == COMPARE_VAL)
			countCPU++;
	}

	printf("The count as calculated in CPU: %d\n", countCPU);

	assert(countCPU == output);

	free(dataArray);

	//CUT_EXIT(argc, argv);

	return 0;
}

