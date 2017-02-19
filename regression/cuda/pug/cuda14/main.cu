#include <call_kernel.h>
#include <stdio.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define MATRIX_X 256
#define MATRIX_Y 128
#define GRID_X 4
#define GRID_Y 4
#define BLOCK_X 1
#define BLOCK_Y 64


__host__ void AddOnDevice (int *h_array_a, int *h_array_b, int *h_array_out, int l_array, int amount_one_work);


int main (int argc, char ** argv) {
	int i, j;
	int l_array = MATRIX_X * MATRIX_Y;		// the length of the in/out arrays
	int *array_a, *array_b, *array_out;
	int amount_one_work = l_array / (GRID_X*GRID_Y*BLOCK_X*BLOCK_Y);
	char forEnd;
	FILE * outfile;


	// malloc the array_a and array_b and array_out
	array_a = (int *) malloc(sizeof(int)*l_array);
	array_b = (int *) malloc(sizeof(int)*l_array);
	array_out = (int *) malloc(sizeof(int)*l_array);


	// fill out the input arrays
	for (i = 0 ; i < l_array ; i++) {
		array_a[i] = rand() % 10;
		array_b[i] = rand() % 10;
		array_out[i] = 0;
	}


	// start to add
	AddOnDevice(array_a, array_b, array_out, l_array, amount_one_work);

	// print out the result
	outfile = fopen("./Assignment1_Problem2_output.txt", "w");
	for (i = 0 ; i < MATRIX_Y ; i++) {
		for (j = 0 ; j < MATRIX_X ; j++) {
			printf("a[%d][%d] (%d) + b[%d][%d] (%d) = out[%d][%d] (%d)\n", i, j, array_a[i*MATRIX_X+j], i, j, array_b[i*MATRIX_X+j], i, j, array_out[i*MATRIX_X+j]);
			fprintf(outfile, "a[%d][%d] (%d) + b[%d][%d] (%d) = out[%d][%d] (%d)\n", i, j, array_a[i*MATRIX_X+j], i, j, array_b[i*MATRIX_X+j], i, j, array_out[i*MATRIX_X+j]);
			assert(array_a[i*MATRIX_X+j] + array_b[i*MATRIX_X+j] == array_out[i*MATRIX_X+j]);
		}
	}
	fclose(outfile);


	printf("Press any key to end\n");
	scanf("%c", &forEnd);
	return 0;
}


//
__global__ void AddMatrix (int *d_array_a, int *d_array_b, int *d_array_out, int amount_one_work) {
	int i;
	int my_tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
	int my_start_x = my_tid_in_block + blockIdx.x * blockDim.x * blockDim.y;
	int my_start_y = blockIdx.y * amount_one_work;
	int step = gridDim.x * blockDim.x * blockDim.y;
	int now_point = my_start_x + my_start_y * step;			// this is the start point in the "array like matrix"


	for (i = 0 ; i < amount_one_work ; i++) {
		d_array_out[now_point] = d_array_a[now_point] + d_array_b[now_point];
		now_point += step;
	}
}


//
__host__ void AddOnDevice (int *h_array_a, int *h_array_b, int *h_array_out, int l_array, int amount_one_work) {
	int *d_array_a, *d_array_b, *d_array_out;
	int i;


	dim3 dimBlock(BLOCK_X, BLOCK_Y);
	dim3 dimGrid(GRID_X, GRID_Y);


	// malloc on device
	//
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_array_a, sizeof(int)*l_array));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_array_b, sizeof(int)*l_array));
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_array_out, sizeof(int)*l_array));


	// copy data to dvice
	CUDA_SAFE_CALL(cudaMemcpy(d_array_a, h_array_a, sizeof(int)*l_array, cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_array_b, h_array_b, sizeof(int)*l_array, cudaMemcpyHostToDevice));


	// create threads for add sum

	AddMatrix<<<dimGrid, dimBlock>>>(d_array_a, d_array_b, d_array_out, amount_one_work);


	// copy back the result
	CUDA_SAFE_CALL(cudaMemcpy(h_array_out, d_array_out, sizeof(int)*l_array, cudaMemcpyDeviceToHost));


	// free memory
	CUDA_SAFE_CALL(cudaFree(d_array_a));
	CUDA_SAFE_CALL(cudaFree(d_array_b));
	CUDA_SAFE_CALL(cudaFree(d_array_out));
}

