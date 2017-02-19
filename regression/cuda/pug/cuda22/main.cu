#include <call_kernel.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------

#define	BLOCKS_IN_X			4
#define	BLOCKS_IN_Y			4
#define THREADS				64
#define THREADS_IN_BLOCK	64
#define THREAD_ACCESSES		32

#define	COLS				(BLOCKS_IN_X * THREADS_IN_BLOCK)
#define	ROWS				(BLOCKS_IN_Y * THREAD_ACCESSES)
#define SIZE				(COLS * ROWS)

#define DEBUG

//-----------------------------------------------------------------------------
// Function Prototypes
//-----------------------------------------------------------------------------

__global__	void	add_matrix_gpu (int* a, int* b, int* c);
__host__	void	calculate (int* h_a, int* h_b, int* h_c, int* h_cpu);
__host__	void	initializeArrays (int* h_a, int* h_b, int* h_c, int* h_sum);
__host__	void	printArray (int* h_a, int* h_b, int* h_c, int* h_sum);

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

__global__ void add_matrix_gpu(int* a, int* b, int* c)
{
	int	index;

	index = blockIdx.y * 32 * COLS +
			blockIdx.x * 64 + threadIdx.x;

	for (int i = 0; i < THREAD_ACCESSES; i++)
	{
		c[index] = a[index] + b[index];

		index += COLS;
	}
}

void calculate (int* h_a, int* h_b, int* h_c, int* h_cpu)
{
	int*	d_a;
	int*	d_b;
	int*	d_c;
	dim3	dimGrid(4, 4);

	// Allocate memory for device copies
	cudaMalloc((void **) &d_a, SIZE * sizeof(int));
	cudaMalloc((void **) &d_b, SIZE * sizeof(int));
	cudaMalloc((void **) &d_c, SIZE * sizeof(int));

	// Copy over the arrays
	cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// Sum the arrays
	add_matrix_gpu <<<dimGrid, THREADS, 0>>> (d_a, d_b, d_c);

	// Copy the summation back
	cudaMemcpy(h_c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__host__ void initializeArrays (int* h_a, int* h_b, int* h_c, int* h_sum)
{
	// Initialize host copies
	srand(0);

	for (int i = 0; i < SIZE; i++)
	{
		h_a[i] 	= rand()%10;
		h_b[i] 	= rand()%10;
		h_sum[i]= h_a[i] + h_b[i];
	}
}

__host__ void printArrays (int* h_a, int* h_b, int* h_c, int* h_sum)
{
	FILE*	fp;

	fp = fopen("results.txt", "w");

	if (fp != NULL)
	{
		fprintf(fp, "Input A Input B CPU Sum GPU Sum\n");
		fprintf(fp, "===============================\n");

		for (int i = 0; i < SIZE; i++)
		{
			fprintf(fp, "%7d %7d %7d %7d\n", h_a[i], h_b[i], h_sum[i], h_c[i]);
		}
	}

	fclose(fp);
}

//-----------------------------------------------------------------------------
// Main Function
//-----------------------------------------------------------------------------

int main (int argc, char** argv)
{
	int*	h_a		= (int*) malloc(SIZE * sizeof(int));
	int*	h_b		= (int*) malloc(SIZE * sizeof(int));
	int*	h_c		= (int*) malloc(SIZE * sizeof(int));
	int*	h_sum	= (int*) malloc(SIZE * sizeof(int));

	initializeArrays(h_a, h_b, h_c, h_sum);
	calculate(h_a, h_b, h_c, h_sum);
	printArrays(h_a, h_b, h_c, h_sum);

#if defined(DEBUG) || defined(_DEBUG)
	for (int i = 0; i < SIZE; i++)
	{
		if (h_c[i] != h_sum[i])
		{
			printf ("ERROR: Element %d does not match\n", i);
		}
	}
#endif

	for (int i = 0; i < SIZE; i++)
		assert(h_c[i]==h_sum[i]);

	free(h_a);
	free(h_b);
	free(h_c);
	free(h_sum);
}

