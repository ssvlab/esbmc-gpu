#include <call_kernel.h>
//pass
//--blockDim=2 --gridDim=2

#include <stdio.h>

#define DIM 2
#define N DIM*DIM

struct S {
  int * p;
};

__global__ void foo(struct S A) {

	A.p[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x; // OBSERVAR AQUI Q É O 'p' Q TEM QUATRO POSIÇÕES E NAO O 'A'

}

int main() {
	S host_A;
	int host_p[N] = {0,0,0,0};

	/* it is not necessary a
	 *  S dev_A 				*/
	int* dev_p;
	int size = N*sizeof(int);

	// 1. Allocate device array.
	cudaMalloc((void**) &(dev_p), size);

	// 2. Copy array contents from host to device.
	cudaMemcpy(dev_p, host_p, size, cudaMemcpyHostToDevice);

	// 3. Point to device pointer in host struct.
		host_A.p = dev_p;

	// 4. Call kernel with host struct as argument
	foo<<<DIM,DIM>>>(host_A);

	// 5. Copy pointer from device to host.
	cudaMemcpy(host_p, dev_p, sizeof(int)*N, cudaMemcpyDeviceToHost);

	// 6. Point to host pointer in host struct
	//    (or do something else with it if this is not needed)
	host_A.p = host_p;

	for(int i = 0; i < N; ++i)
		printf("%d	", host_p[i]);

	cudaFree(&dev_p);
	return 0;
}
