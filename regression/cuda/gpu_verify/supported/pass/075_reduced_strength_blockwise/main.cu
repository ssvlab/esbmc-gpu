//pass
//--blockDim=256 --gridDim=2 -DWIDTH=2064 --no-inline
#include <call_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define GRIDDIM 1
#define BLOCKDIM 2//256
#define WIDTH 2//2048
#define N WIDTH
/*
 * This kernel demonstrates a blockwise strength-reduction loop.
 * Each block is given a disjoint partition (of length WIDTH) of A.
 * Then each thread writes multiple elements in the partition.
 * It is not necessarily the case that WIDTH%blockDim.x == 0
 */

__global__ void k(int *A) {
//  __assert(blockDim.x <= WIDTH);
//#ifdef BLOCK_DIVIDES_WIDTH
//  //__assert(__mod_pow2(WIDTH, blockDim.x) == 0);
//#endif

  for (int i=threadIdx.x; i<WIDTH; i+=blockDim.x) {

//#ifndef BLOCK_DIVIDES_WIDTH
//    // working set(1) using global invariants
//    /*A*/__global_invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset_bytes(A)/sizeof(int))),
//    /*B*/__global_invariant(__write_implies(A,                       __write_offset_bytes(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
//    /*C*/__invariant(threadIdx.x <= i),
//    /*D*/__invariant(               i <= WIDTH+blockDim.x),
//         __invariant(i % blockDim.x == threadIdx.x),
//         __global_invariant(__write_implies(A, (((__write_offset_bytes(A)/sizeof(int)) % WIDTH) % blockDim.x) == threadIdx.x)),
//#else
//    // working set(2) iff WIDTH % blockDim.x == 0
//    /*A*/__invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset_bytes(A)/sizeof(int))),
//    /*B*/__invariant(__write_implies(A,                       __write_offset_bytes(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
//    /*C*/__invariant(threadIdx.x <= i),
//    /*D*/__invariant(               i <= WIDTH+blockDim.x),
//         __invariant(__uniform_int((i-threadIdx.x))),
//         __invariant(__uniform_bool(__enabled())),
//#endif

    A[blockIdx.x*WIDTH+i] = i;
  }

//#ifdef FORCE_FAIL
//  __assert(false);
//#endif
}

int main (){
	int *a;
	int *dev_a;
	int size = N*sizeof(int);

	cudaMalloc((void**)&dev_a, size);

	a = (int*)malloc(size);

	for (int i = 0; i < N; i++)
		a[i] = 0;

//	printf("Old a:  ");
//	for (int i = 0; i < N; i++)
//		printf("%d	", a[i]);

	cudaMemcpy(dev_a,a,size,cudaMemcpyHostToDevice);

	//k <<<GRIDDIM, BLOCKDIM>>>(dev_a);
	ESBMC_verify_kernel(k,GRIDDIM,BLOCKDIM,dev_a);

	cudaMemcpy(a,dev_a,size,cudaMemcpyDeviceToHost);

//	printf("\nNew a:  ");

	for (int i = 0; i < N; i++){
//		printf("%d	", a[i]);
		assert(a[i]== i);
	}

	free(a);
	cudaFree(dev_a);
	return 0;
}
