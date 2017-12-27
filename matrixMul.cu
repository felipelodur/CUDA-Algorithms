#include    "../../include/wb.h"

#define wbCheck(stmt) do {                                 
        cudaError_t err = stmt;                            
        if (err != cudaSuccess) {                          
            wbLog(ERROR, "Failed to run stmt ", #stmt);    
            return -1;                                     
        }                                                  
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here

        if ((numARows < 256) && (numBColumns < 256)) {
        float Pvalue = 0;
                for (int k = 0; k < 256; ++k) Pvalue += A[numARows * 256 + k] * B[ k * 256 + numBColumns];
        C[numARows * 256 + numBColumns] = Pvalue;
        }

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    hostC = (float *) wbImport(wbArg_getInputFile(args, 2), &numCRows, &numCColumns);

//    numCRows = 0;
//    numCColumns = 0;

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");

    // Allocate GPU memory
    cudaMalloc((void **)&deviceA, numARows*numAColumns*sizeof(float));
    cudaMalloc((void **)&deviceB, numBRows*numBColumns*sizeof(float));
    cudaMalloc((void **)&deviceC, numCRows*numCColumns*sizeof(float));


    wbTime_stop(GPU, "Allocating GPU memory.");

   wbTime_start(GPU, "Copying input memory to the GPU.");
    // Copy memory to the GPU
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    // Initialize the grid and block dimensions
    dim3 dimGrid(numCRows/256, numCRows/256,1);
    dim3 dimBlock(256, 256, 1);

    wbTime_start(Compute, "Performing CUDA computation");

    // Launch the GPU Kernel
    matrixMultiply <<<dimGrid, dimBlock>>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    // Copy memory back
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    wbTime_stop(GPU, "Freeing GPU Memory");
wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}



