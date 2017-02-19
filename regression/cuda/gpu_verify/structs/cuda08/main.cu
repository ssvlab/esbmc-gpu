#include <call_kernel.h>
/**************************************** align.cu **************************************/

//#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <assert.h>
#define N 4	/*condição suficiente: números pares (N = quantidade de threads lançadas)*/

/*estrutura 'pair' para guardar o índice do bloco e da thread de determinada thread*/
typedef struct __align__(64) {

	unsigned int tid, bid;

} pair;										/*pair é o 'apelido' da estrutura sem nome*/

__global__ void* align_test (pair* A) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int idx = blockDim.x * bid + tid;

	A[idx].tid = tid;
	A[idx].bid = bid;
}

int main()
{
	pair* dev_b;
	pair* b;

	b = (pair*)malloc(N*sizeof(pair)); /* acessível apenas pela CPU função main e funções __host__ */

	cudaMalloc((void**)&dev_b, N*sizeof(pair)); /* acessível apenas pela GPU funções __global__ */

	//align_test<<<N/2, 2>>>(dev_b);
	ESBMC_verify_kernel(align_test,N/2,2,dev_b);

	cudaMemcpy(b, dev_b, N*sizeof(pair), cudaMemcpyDeviceToHost);

	printf("\ttid\tbid\n");
	for (int i = 0; i < N; ++i){
	   printf("b[%d]: \t %d\t %d\n", i, b[i].bid, b[i].tid);
	   if(i%2==0)
		   assert((b[i].bid==i/2)and(b[i].tid==0));
	   else
		   assert((b[i].bid==i/2)and(b[i].tid==1));
	}

	cudaFree(dev_b);

    return 0;
}

