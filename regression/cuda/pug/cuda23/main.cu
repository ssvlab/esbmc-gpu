#include <call_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#define NUM_OF_ROWS 128
#define NUM_OF_ROWS_PER_BLOCK 32
#define NUM_OF_COLS 256
#define NUM_OF_COLS_PER_BLOCK 64

#define BLOCKSIZE 4

typedef int TYPE;

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
/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ void add_matrix_gpu(TYPE *a,TYPE *b, TYPE *c)
	{
	int x = blockIdx.x* blockDim.x * NUM_OF_ROWS_PER_BLOCK + threadIdx.x * NUM_OF_ROWS_PER_BLOCK;
	//blockDim.y == 4
	int y = blockIdx.y * NUM_OF_ROWS_PER_BLOCK * NUM_OF_COLS;
	int index = 0;
	for (int i=0; i<NUM_OF_ROWS_PER_BLOCK; i++){
		index = x + y + i;
		c[index] = a[index] + b[index];
	}
}

/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{

	if(!InitCUDA()) {
		return 0;
	}



	unsigned int timer = 0;


	TYPE *local_a = (TYPE*)malloc(NUM_OF_ROWS * NUM_OF_COLS * sizeof(TYPE));
	TYPE *local_b = (TYPE*)malloc(NUM_OF_ROWS * NUM_OF_COLS * sizeof(TYPE));
	TYPE *local_out = (TYPE*)malloc(NUM_OF_ROWS * NUM_OF_COLS * sizeof(TYPE));
	for (int i=0; i<NUM_OF_ROWS; i++){
		for (int j=0; j<NUM_OF_COLS; j++){
			local_a[i * NUM_OF_ROWS + j] = (rand() %10);
			local_b[i *NUM_OF_ROWS + j] = (rand() %10);
			local_out[i *NUM_OF_ROWS + j] = 0;

		}
	}
	//for (int i=0; i<256; i++){
	//	printf("%d\n", local_out[i]);
	//}
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	TYPE *a,*b,*c;

	cudaMalloc((void **) &a,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE));
	cudaMalloc((void **) &b,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE));
	cudaMalloc((void **) &c,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE));
//	cudaMemset(d_out_array, 0,INIT_BLOCKSIZE * sizeof(int));
	cudaMemcpy(a,local_a,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE),cudaMemcpyHostToDevice);
	cudaMemcpy(b,local_b,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE),cudaMemcpyHostToDevice);
	cudaMemset(c,0, NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE));
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	add_matrix_gpu<<<dimBlock,64>>>(a,b,c);

	CUT_CHECK_ERROR("Kernel execution failed\n");

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_SAFE_CALL( cutStopTimer( timer));

	FILE *fp;
	fp = fopen("readme2.txt", "w");

	fprintf(fp, "Processing time: %f (ms)\n", cutGetTimerValue( timer));
	CUT_SAFE_CALL( cutDeleteTimer( timer));


	cudaMemcpy(local_out,c,NUM_OF_ROWS * NUM_OF_COLS*sizeof(TYPE),cudaMemcpyDeviceToHost);
	int counter = 0;
	for (int i=0; i<32768; i++){
			fprintf(fp, "%d %d %d %d\n", i, local_a[i], local_b[i], local_out[i]);
		if (local_a[i] + local_b[i] != local_out[i]){
			counter++;
		}
	}
	fprintf(fp, "Number incorrect: %d\n",counter);

	fclose(fp);
	CUT_EXIT(argc, argv);

	for (int i=0; i<NUM_OF_ROWS; i++){
		for (int j=0; j<NUM_OF_COLS; j++){
			assert(local_out[i *NUM_OF_ROWS + j] == local_a[i * NUM_OF_ROWS + j] + local_b[i *NUM_OF_ROWS + j]);
		}
	}

	getchar();
	return 0;
}

