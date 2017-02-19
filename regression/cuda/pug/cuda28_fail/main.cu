#include <call_kernel.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>




#define SIZE 32768
#define BLOCKSIZE 16

__host__ void outer_compute(int *in_arr, int *m_arr,int *out_arr);




__global__ void compute(int *d_in,int *d_m_arr,int *d_out) {
  int i;


int bid=blockIdx.x+blockIdx.y*4;

for (i=bid*(SIZE/BLOCKSIZE)+threadIdx.x*32;i<(bid*(SIZE/BLOCKSIZE)+((threadIdx.x+1)*32));i++)
{
	if( i <SIZE )
	d_out[i]=d_in[i]+d_m_arr[i];
}


}




__host__ void outer_compute(int *h_in_array,int*h_m_array, int *h_out_array) {
  int *d_in_array, *d_out_array,*d_m_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_m_array,SIZE*sizeof(int));

  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_array,h_out_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_m_array,h_m_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);

 dim3 dimBlock(64,1);
 dim3 dimGrid(4,4);
  compute<<<dimGrid,dimBlock,0>>>(d_in_array,d_m_array,d_out_array);
  cudaMemcpy(h_out_array,d_out_array,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

}



int main(int argc, char **argv)
{
  int *in_array,*m_array, *out_array,*temp_array;
  int sumc=0;int sumg=0;

 /* initialization */

  in_array = (int *) malloc(SIZE*sizeof(int));
  m_array = (int *) malloc(SIZE*sizeof(int));
  out_array = (int *) malloc(SIZE*sizeof(int));
  temp_array = (int *) malloc(SIZE*sizeof(int));

  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
	 m_array[i] = rand()%10;

	 temp_array[i]=in_array[i]+m_array[i];
  }




  /* compute matrix addition of in_array and m_array*/
  outer_compute(in_array,m_array, out_array);

  for (int i=0; i<SIZE; i++) {
	 // printf("%d  ",out_array[i]);
	 //  printf("%d \n",temp_array[i]);
	 //  system("PAUSE");
    sumg+=out_array[i];
	sumc+=temp_array[i];
  }

printf ("total Sum counted on CPU : %d \n" ,sumc);
printf ("total Sum counted on GPU : %d \n" ,sumg);

	assert(sumc != sumg);

//system("PAUSE");
}

