#include <call_kernel.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#define SIZE 256
#define BLOCKSIZE 32


__host__ void outer_compute(int *in_arr, int *out_arr);
__host__ void matrix_compute(int *hA, int *hB, int *hC);


int N=32*4;
int M=64*4;
__host__ void printArray(int* A,int* B,int* C){
	FILE * pFile;
	pFile = fopen("output.txt","w");

	fprintf(pFile,"A\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",A[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");

	fprintf(pFile,"B\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",B[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");

	fprintf(pFile,"C\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			fprintf(pFile,"%d\t",C[i*N+j]);
		}
		fprintf(pFile,"\n");
	}
	fprintf(pFile,"\n");
	fclose (pFile);
}
__global__ void deviceMatCompute(int *A,int *B, int *C,int N,int M) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;



	for(int i=0; i<32; i++){
		int index = M*by*32+bx*64+tx+i*M;
		C[index] = A[index] + B[index];
	__syncthreads();
	}
}
__host__ void matrix_compute(int *hA, int *hB, int *hC) {
	int *dA,*dB,*dC;

	/* allocate memory for device copies, and copy input to device */
	cudaMalloc((void **) &dA,N*M*sizeof(int));
	cudaMalloc((void **) &dB,N*M*sizeof(int));
	cudaMalloc((void **) &dC,N*M*sizeof(int));
	cudaMemcpy(dA,hA,N*M*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dB,hB,N*M*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dC,hC,N*M*sizeof(int),cudaMemcpyHostToDevice);
	dim3 grid(4,4);
	dim3 block(64);


	/* compute number of appearances of 8 for subset of data in each thread! */
	deviceMatCompute<<<grid,block,0>>>(dA,dB,dC,N,M);

	cudaMemcpy(hA,dA,N*M*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(hB,dB,N*M*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(hC,dC,N*M*sizeof(int),cudaMemcpyDeviceToHost);
}

void matrixDeviceSide(){
	int* A,*B,*C;
	N = 32*4;
	M = 64*4;
	//allocate space for original matricies
	A = (int*)malloc(sizeof(int)*N*M);
	B = (int*)malloc(sizeof(int)*N*M);
	C = (int*)malloc(sizeof(int)*N*M);

	//fill in matricies

	//A
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			A[i*N+j] = rand()%10;
		}
	}

	//B
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			B[i*N+j] = rand()%10;
		}
	}

	//C
	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++){
			C[i*N+j] = 0;
		}
	}

	matrix_compute(A,B,C);
	printArray(A, B, C);
}
int main(int argc, char **argv)
{
	int matrixOrList = 0;						//change this to 0 for list sum
	if(matrixOrList){
		matrixDeviceSide();
	}
	else{
		int *in_array, *out_array;

		/* initialization */
		int counter = 0;
		in_array = (int *) malloc(SIZE*sizeof(int));
		for (int i=0; i<SIZE; i++) {
			in_array[i] = rand()%10;
			printf("in_array[%d] = %d",i,in_array[i]);
			if(in_array[i] == 6){
				counter++;
				printf("     <------  %d",counter);
			}
			printf("\n");
		}
		int sum = 0;
		int sum_cpu = 0;
		for(int i=0; i<SIZE; i++){
			sum += in_array[i];
			printf("%d; ", sum);
			if (in_array[i] == 6)
				sum_cpu++;
		}
		printf("\n");
		out_array = (int *) malloc(SIZE*sizeof(int));

		/* compute number of appearances of 6 */
		outer_compute(in_array, out_array);

		for (int i=0; i<BLOCKSIZE; i++) {
		sum=out_array[0];
		}

		printf ("The number 6 appears %d times in array of  %d numbers\n",sum,SIZE);
		assert(sum_cpu == sum);
	}
}
__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}
__global__ void compute(int *d_in,int *d_out) {
	int i;

	d_out[threadIdx.x] = 0;
	for (i=0; i<SIZE/BLOCKSIZE; i++) {
		d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
	}
		__syncthreads();
	for(int i=2,j=1; i<SIZE/4; i*=2, j*=2){
		if(threadIdx.x%i == 0){
			d_out[threadIdx.x] += d_out[threadIdx.x + j];
			//printf ("d_out[%d]=%d; ",threadIdx.x, d_out[threadIdx.x]);
		}
		__syncthreads();
	}
	__syncthreads();
}

__global__ void compute(int *A,int *B, int *C) {

	//C[BlockIdx.x*N + threadIdx.x] = 0;
	for (int i=0; i<SIZE/BLOCKSIZE; i++) {
		//d_out[threadIdx.x] += compare(d_in[i*BLOCKSIZE+threadIdx.x],6);
	}
		__syncthreads();
	for(int i=2,j=1; i<SIZE; i*=2, j*=2){
		if(threadIdx.x%i == 0){
			//d_out[threadIdx.x] += d_out[threadIdx.x + j];
		}
		__syncthreads();
	}
	__syncthreads();
}
__host__ void outer_compute(int *h_in_array, int *h_out_array) {
	int *d_in_array, *d_out_array;

	/* allocate memory for device copies, and copy input to device */
	cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
	cudaMalloc((void **) &d_out_array,SIZE*sizeof(int));
	cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	printf("\n");

	/* compute number of appearances of 8 for subset of data in each thread! */
	compute<<<1,BLOCKSIZE,(SIZE+BLOCKSIZE)*sizeof(int)>>>(d_in_array,d_out_array);

	cudaMemcpy(h_out_array,d_out_array,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
	int sum = h_out_array[0];
	int i=0;
}

