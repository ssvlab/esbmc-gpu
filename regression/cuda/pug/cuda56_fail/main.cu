#include <call_kernel.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <assert.h>


#define SIZE 256
#define BLOCKSIZE 32

__host__ void outer_compute(int *in_arr, int *out_arr);

__device__ int compare(int a, int b) {
  if (a == b) return 1;
  return 0;
}




__global__ void compute(int *d_in,int *d_out) {
  int i;


  d_out[threadIdx.x] = 0;
  for (i=0; i<SIZE/BLOCKSIZE; i++) {
      d_out[threadIdx.x] += compare(d_in[(SIZE/BLOCKSIZE)*threadIdx.x+i],6);
  }

__syncthreads();


int base=2;
/* number of  times the loop should run or Depth or tree
   is log2 of the BLOCKSIZE
*/
int count=5;

while(count){

if (threadIdx.x%base==0)
{
	d_out[threadIdx.x]+=d_out[threadIdx.x+(base/2)];

}
__syncthreads();


count--;
base*=2;

}


}

__host__ void outer_compute(int *h_in_array, int *h_out_array) {
  int *d_in_array, *d_out_array;

  /* allocate memory for device copies, and copy input to device */
  cudaMalloc((void **) &d_in_array,SIZE*sizeof(int));
  cudaMalloc((void **) &d_out_array,BLOCKSIZE*sizeof(int));

  cudaMemcpy(d_in_array,h_in_array,SIZE*sizeof(int),cudaMemcpyHostToDevice);


  /* compute number of appearances of 8 for subset of data in each thread! */
  compute<<<1,BLOCKSIZE>>>(d_in_array,d_out_array);
  cudaMemcpy(h_out_array,d_out_array,BLOCKSIZE*sizeof(int),cudaMemcpyDeviceToHost);

  /*code printing the final output array ,uncomment for testing
  for (int i=0;i<BLOCKSIZE;i++)
  {
	  printf("%d \n" ,h_out_array[i]);
  }
  */
	printf ("Total occurences of 6 as counted on GPU: %d",h_out_array[0]);

}



int main(int argc, char **argv)
{
  int *in_array, *out_array;
  int sum=0;
int s=0;

 /* initialization */
  in_array = (int *) malloc(SIZE*sizeof(int));
  for (int i=0; i<SIZE; i++) {
    in_array[i] = rand()%10;
    printf("in_array[%d] = %d\n",i,in_array[i]);

/* Counting the Actual number of 6's to ESBMC_verify answer */
 if(in_array[i]==6)
  {
	  s++;}

  }

  printf ("total 6's counted on CPU : %d \n" ,s);

  out_array = (int *) malloc(BLOCKSIZE*sizeof(int));

  /* compute number of appearances of 6 */
  outer_compute(in_array, out_array);

  for (int i=0; i<BLOCKSIZE; i++) {
    sum+=out_array[i];
  }
  assert(out_array[0] != s);

 //system("PAUSE");
}

