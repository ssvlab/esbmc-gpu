#include <call_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

/*
 * These are the parameters that determine what elements the different
 * threads operate on.
 */
#define GRID_SIZE           4   /* Blocks per row and column of grid. */
#define ELEMENTS_PER_THREAD 32  /* Number of matrix elements each thread adds. */
#define THREADS_PER_COLUMN  1   /* Number of threads operating on column in block (this * ELEMENTS_PER_THREAD is block row count). */
#define THREADS_PER_ROW     64  /* Number of threads operating on row in block (block column count). */

/* Helper definitions based on the above data - don't change. */
#define THREADS_PER_BLOCK   (THREADS_PER_COLUMN*THREADS_PER_ROW)
#define MATRIX_WIDTH        (GRID_SIZE*THREADS_PER_ROW)
#define MATRIX_HEIGHT       (GRID_SIZE*THREADS_PER_COLUMN*ELEMENTS_PER_THREAD)
#define MATRIX_ELEMENTS     (MATRIX_WIDTH*MATRIX_HEIGHT)

/* Function prototypes. */
__host__ void print_matrix(int *matrix);
__host__ void add_matrix(int *a, int *b, int *c);
__global__ void add_matrix_gpu(int *a, int *b, int *c);

/*
 * main
 * This is the main function that initializes the device, sets up the two
 * random matrices of integers to add, and dispatches to the device to add
 * the matrices. In debug mode, the result is also checked against the CPU
 * computed data.
 */
int main(int argc, char **argv)
{
    int *Amatrix, *Bmatrix, *Cmatrix;
    int i;

    /* Initializes the CUDA device. */
    CUT_DEVICE_INIT(argc, argv);

    /* Initializes the matricies with random numbers. */
    Amatrix = (int *) malloc(MATRIX_ELEMENTS * sizeof(int));
    Bmatrix = (int *) malloc(MATRIX_ELEMENTS * sizeof(int));
    Cmatrix = (int *) malloc(MATRIX_ELEMENTS * sizeof(int));
    if (Amatrix == NULL || Bmatrix == NULL || Cmatrix == NULL)
    {
        fprintf(stderr, "Allocation of matrix memory failed!");
        return 1;
    }
    for (i = 0; i < MATRIX_ELEMENTS; i++)
    {
        Amatrix[i] = rand() % 10;
        Bmatrix[i] = rand() % 10;
        Cmatrix[i] = 0;
    }

    /* Print the stats. */
    printf("Problem Stats\n");
    printf("--------------\n");
    printf("Matrix width: %d\n", MATRIX_WIDTH);
    printf("Matrix height: %d\n", MATRIX_HEIGHT);
    printf("Matrix elements: %d\n", MATRIX_ELEMENTS);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Threads access %d elements\n", ELEMENTS_PER_THREAD);
    printf("Block arrangement in grid: %dx%d\n", GRID_SIZE, GRID_SIZE);
    printf("\n");

    /* Print out the matricies. */
    printf("Matrix A\n");
    printf("--------------\n");
    print_matrix(Amatrix);
    printf("\n");
    printf("Matrix B\n");
    printf("--------------\n");
    print_matrix(Bmatrix);
    printf("\n");

    /* Compute the matrix sums (on the GPU). */
    add_matrix(Amatrix, Bmatrix, Cmatrix);

    /* Print out the resulting matrix. */
    printf("Sum Result\n");
    printf("--------------\n");
    print_matrix(Cmatrix);
    printf("\n");

#ifdef _DEBUG
    /* Check the result. */
    bool allValid = true;
    for (i = 0; i < MATRIX_ELEMENTS; i++)
    {
        if (Amatrix[i] + Bmatrix[i] != Cmatrix[i])
        {
            printf("Error: Element %d,%d of result is invalid.\n", i % MATRIX_WIDTH, i / MATRIX_WIDTH);
            allValid = false;
        }
    }
    if (allValid)
        printf("The matrix addition was checked successfully.\n");
#endif

   for (i = 0; i < MATRIX_ELEMENTS; i++){
	   assert(Cmatrix[i] == Amatrix[i] + Bmatrix[i]);
   }
    /* Free the allocated memory. */
    free(Amatrix);
    free(Bmatrix);
    free(Cmatrix);

    /* Clean up and exit. */
    CUT_EXIT(argc, argv);
    return 0;
}

/*
 * print_matrix
 * Prints the given matrix to standard output. It is useful for debugging on
 * small matrices.
 */
__host__ void print_matrix(int *matrix)
{
    int i;

    for (i = 0; i < MATRIX_ELEMENTS; i++)
    {
        if (i != 0 && (i % MATRIX_WIDTH) == 0)
            printf("\n");
        printf("%-3d", matrix[i]);
    }
    printf("\n");
}

/*
 * add_matrix
 * This function initializes the memory on the device, copies the input
 * matrices A and B to the device, calls the kernel to compute the matrix sums,
 * and copies the result back in to the C variable.
 */
__host__ void add_matrix(int *A, int *B, int *C)
{
    int *d_Amatrix, *d_Bmatrix, *d_Cmatrix;

    /* Grid size of 16 blocks (4x4 arrangement). */
    dim3 dimGrid(GRID_SIZE, GRID_SIZE);

    /* Block size of 64 threads. Each block operates on a 64x32 area. */
    dim3 dimBlock(THREADS_PER_ROW, THREADS_PER_COLUMN);

    /* Allocate the memory for the three matrices, and copy A and B over. */
    CUDA_SAFE_CALL( cudaMalloc((void **) &d_Amatrix, MATRIX_ELEMENTS * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc((void **) &d_Bmatrix, MATRIX_ELEMENTS * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc((void **) &d_Cmatrix, MATRIX_ELEMENTS * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Amatrix, A, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Bmatrix, B, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice) );

    /* Call upon the GPU to compute the matrix sum. */
    //add_matrix_gpu<<<dimGrid, dimBlock>>>(d_Amatrix, d_Bmatrix, d_Cmatrix);
    ESBMC_verify_kernel_with_three_args(add_matrix_gpu, dimBlock, dimGrid, d_Amatrix, d_Bmatrix, d_Cmatrix);

    CUT_CHECK_ERROR("The GPU computation function failed!\n");

    /* Copy the computed matrix back. */
    CUDA_SAFE_CALL( cudaMemcpy(C, d_Cmatrix, MATRIX_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );

    /* Free the memory on the device. */
    CUDA_SAFE_CALL( cudaFree(d_Amatrix) );
    CUDA_SAFE_CALL( cudaFree(d_Bmatrix) );
    CUDA_SAFE_CALL( cudaFree(d_Cmatrix) );
}

/*
 * add_matrix_gpu
 * This function is ESBMC_executed on the device. The input matrices are split in to
 * sections, so each thread adds ELEMENTS_PER_THREAD elements.
 */
__global__ void add_matrix_gpu(int *A, int *B, int *C)
{
    int x, y;
    int index, end;

    /* Calculate the starting position. */
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * ELEMENTS_PER_THREAD * THREADS_PER_COLUMN + ELEMENTS_PER_THREAD * threadIdx.y;
    index = x + y * MATRIX_WIDTH;

    /* Compute the sums. */
    end = index + ELEMENTS_PER_THREAD * MATRIX_WIDTH;
    for (; index < end; index += MATRIX_WIDTH)
    {
        C[index] = A[index] + B[index];
    }
}

