//pass
//blockDim=1024 --gridDim=1 --no-inline

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

typedef double(*funcType)(double);

__device__ double bar(double x) {
  return sin(x);
}

__global__ void foo(double x, int i)
{
  funcType f;

  if (i == 0)
    f = bar;
  else
    f = cos;

  double z = f(x);
	assert(z != NULL);

  printf("z: %f ", z);
}

int main(){

	int select_function = 1; // 1= sen; 0=cos
	double angle = 1.57; //0;

	//foo <<<1,2>>>(angle, select_function);
	ESBMC_verify_kernel_c (foo,1,2,angle, select_function);

	cudaThreadSynchronize();

	return 0;
}
