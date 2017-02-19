#include <call_kernel.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 4

int n = 20; //it defines the range of the random number
using namespace std;

__device__ float generate( curandState* globalState, int ind ) // ind varies from 0 to N
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState ); // float curand_uniform(curandStateXORWOW_t *state
    											// Return a uniformly distributed float between \p 0.0f and \p 1.0f
    globalState[ind] = localState; // localState received a new value based on its own 'seed'
    return RANDOM;
}

__global__ void setup_seed ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );

    // curand_init generate random numbers that will be stored on state[id]
    // seed must be a random number
}

__global__ void kernel(float* N3, curandState* globalState, int n) // n = 2
{
    // generate random numbers
    for(int i=0;i<N;i++)
    {					// globalState received the 'seeds'
        int k = generate(globalState, i) * (10*N/4); // float generate (curandState* globalState, int ind) is a __device__ function
        while(k > n*n-1) // k >3 THIS WHILE DEFINES THE RANGE OF THE RANDOM NUBER, TO n=2 and N=4 deinine the range (0,3]
//        ESTE WHILE DEFINE O RANGE DO NÚMERO ALEATÓRIO, neste caso o limita a (0,3]
        {
            k-=(n*n-1); // k = k -3
        }
        N3[i] = k; //  10 -> 1; 9 -> 3; 8 -> 2; 7 -> 1; etc
    }
}

int main()
{
    curandState* devStates;
    cudaMalloc ( (void**) &devStates, N*sizeof( curandState ) ); // generate N = 4 seeds

    // setup seeds
	
    //setup_seed <<< 1, N >>> ( devStates, unsigned(time(NULL)) ); // here devStates returns with seeeds (4 seeds)
	

    float N2[N]; // host variable
    float* N3; // device variable

    cudaMalloc((void**) &N3, sizeof(float)*N); // generate N3 of size N=4

//    kernel<<<1,1>>> (N3, devStates, n); // n = 2 (variable int global)
	ESBMC_verify_kernel(kernel,1,1,devStates,n);

    cudaMemcpy(N2, N3, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
    {
        cout<<N2[i]<<endl;
    }

    return 0;
}
