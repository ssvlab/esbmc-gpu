#include <call_kernel.h>
//xfail:BOOGIE_ERROR
//--blockDim=64 --gridDim=64 --no-inline
//
#include "cuda.h"
#define N dim*dim
#define dim 2

__global__ void foo() {

  __shared__ int a;

  a = threadIdx.x;
}

int main(){

	//foo <<<N,N>>> ();
	ESBMC_verify_kernel(foo, dim,dim);

	cudaThreadSynchronize();

}
