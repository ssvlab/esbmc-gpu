#include <call_kernel.h>
//pass
//--blockDim=2 --gridDim=2

#include <stdio.h>

#define DIM 2
#define N DIM*DIM

struct S {
  struct {
    int * p;
    int * q;
  } s;
};

__global__ void foo(struct S A) {

	A.s.p[threadIdx.x + blockDim.x*blockIdx.x] = A.s.q[threadIdx.x + blockDim.x*blockIdx.x] + threadIdx.x;
	A.s.q[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;

}

int main() {
	S host_A;
	int host_p[N] = {0,0,0,0};
	int host_q[N] = {1,1,1,1};

	/* it is not necessary a
	 *  S dev_A 				*/
	int* dev_p;
	int* dev_q;
	int size = N*sizeof(int);

	// 1. Allocate device array.
	cudaMalloc((void**) &(dev_p), size);
	cudaMalloc((void**) &(dev_q), size);

	// 2. Copy array contents from host to device.
	cudaMemcpy(dev_p, host_p, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_q, host_q, size, cudaMemcpyHostToDevice);

	// 3. Point to device pointer in host struct.
		host_A.s.p = dev_p;
		host_A.s.q = dev_q;

	// 4. Call kernel with host struct as argument
	foo<<<DIM,DIM>>>(host_A);

	// 5. Copy pointer from device to host.
	cudaMemcpy(host_p, dev_p, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_q, dev_q, size, cudaMemcpyDeviceToHost);

	// 6. Point to host pointer in host struct
	//    (or do something else with it if this is not needed)
	host_A.s.p = host_p;
	host_A.s.q = host_q;

	printf("\np: ");
	for(int i = 0; i < N; ++i)
		printf("	%d", host_p[i]);

	printf("\nq: ");
	for(int i = 0; i < N; ++i)
		printf("	%d", host_q[i]);
}
