#include <call_kernel.h>
#include<stdio.h>
#include <assert.h>

#define NX 256
#define NZ 128
#define iter 32

__global__ void add_matrix_gpu(int* a,int* b,int* c) {
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   int j=blockIdx.y*blockDim.y+threadIdx.y;

   for(int k=0;k<iter;k++) {
      int index=i+j*NX*iter+k*NX;
      c[index]=a[index]+b[index];
   }

}

FILE *efopen(char *fname,char *type)
{
    FILE *fp;
    if( (fp=fopen(fname,type))==NULL ) {
      printf("open file %s failed.\n",fname);
      exit(1);
    }
    return fp;
}

int main() {

   int n=NX*NZ;
   int msize=n*sizeof(int);

   int *h_A,*h_B,*h_C;

// allocate host memory
   h_A=(int*)malloc(msize);
   h_B=(int*)malloc(msize);
   h_C=(int*)malloc(msize);

// initialize h_A and h_B
   for(int i=0;i<n;i++) {
      h_A[i]=rand()%10;
      h_B[i]=rand()%10;
      h_C[i]=0;
   }

// output initial matrix A and B
   FILE *fpa,*fpb;
   fpa=efopen("A.dat","w");
   fpb=efopen("B.dat","w");
   for(int i=0;i<NZ;i++ ) {
      for(int j=0;j<NX;j++ ) {
         fprintf(fpa,"%d\t", h_A[i*NX+j]);
         fprintf(fpb,"%d\t", h_B[i*NX+j]);
      }
      fprintf(fpa,"\n");
      fprintf(fpb,"\n");
   }
   fclose(fpa);
   fclose(fpb);

// alocate device memory
   int *d_A, *d_B, *d_C;
   cudaMalloc((void**)&d_A,msize);
   cudaMalloc((void**)&d_B,msize);
   cudaMalloc((void**)&d_C,msize);

// copy host memory to device
   cudaMemcpy(d_A,h_A,msize,cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B,msize,cudaMemcpyHostToDevice);

   dim3 dimGrid(4,4);
   dim3 dimBlock(64,1);

   add_matrix_gpu<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);

   cudaMemcpy(h_C,d_C,msize,cudaMemcpyDeviceToHost);

// output matrix C
   FILE *fpc;
   fpc=efopen("C.dat","w");
   for(int i=0;i<NZ;i++ ) {
      for(int j=0;j<NX;j++ ) {
         fprintf(fpc,"%d\t", h_C[i*NX+j]);
      }
      fprintf(fpc,"\n");
   }
   fclose(fpc);

   for(int i=0; i<n; i++)
	   assert(h_C[i] == h_A[i] + h_B[i]);

}

