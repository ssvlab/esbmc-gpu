//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <call_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 512

__global__ void helloCUDA(int x)
{
///    __requires(x == 143);
    __shared__ float S[256*32];
    __shared__ float F[256];

    unsigned int idx;

    //initialise data on shared memory
    for(int i = 0;
   //         __invariant(__implies(__write(S), ((__write_offset_bytes(S)/sizeof(float)) % blockDim.x) == threadIdx.x)),
            i < x;
            i += (blockDim.x/32)) /* translate: i = 0; i < 143; i+=16 , total de iterações: 8*/
		
    {
        if((i+(threadIdx.x/32)) < x){
            idx = (i+(threadIdx.x/32))*32+(threadIdx.x%32);
            S[idx] = F[i+(threadIdx.x/32)];
        }
    }

}
int main (){

	//helloCUDA <<<1,N>>>(143);
	ESBMC_verify_kernel(helloCUDA,1,N, 143);	
	
	cudaThreadSynchronize();

	return 0;
}
