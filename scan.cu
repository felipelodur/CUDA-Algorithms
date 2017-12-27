// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    "../../include/wb.h"

#define BLOCK_SIZE 16


#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
	__shared__ float partialSum[16]; //get rekt
	// recupera indices das threads e blocos
	unsigned int t = threadIdx.x;
	unsigned int b = blockIdx.x*blockDim.x;
	// carrega elemento na memória local
	if(b + t < len)
			partialSum[t] = input[b + t];
	else
			partialSum[t] = 0;
	// barreira de sincronização
	__syncthreads();
	float x;
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if(t >= stride)
			x = partialSum[t] + partialSum[t-stride];
		__syncthreads();
		
		if(t >= stride)
			partialSum[t] = x;
		__syncthreads();
	}
	if(b + t < len)
		input[b+t] = partialSum[t];
	if(t == blockDim.x-1)
		output[blockIdx.x+1] = input[b+t];
}


__global__ void add(float *input, float *output, int len) {
   unsigned int t = threadIdx.x;
   unsigned int b = blockIdx.x*blockDim.x;
   if(b+t < len)
        output[b+t] += input[blockIdx.x];

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list
    int numOutputElements;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    numOutputElements = ((numElements-1)/BLOCK_SIZE+1);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

 wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");

    // Allocate GPU memory
    wbTime_stop(GPU, "Allocating GPU memory.");
    cudaMalloc( (void**)&deviceInput, numElements * sizeof(float) );
    cudaMalloc( (void**)&deviceOutput, numOutputElements * sizeof(float) );

    wbTime_start(GPU, "Copying input memory to the GPU.");
        int inputLength = numElements;

    // Copy data to GPU
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    cudaMemcpy( deviceOutput, hostOutput, numOutputElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( deviceInput, hostOutput, numOutputElements*sizeof(float), cudaMemcpyHostToDevice);


    // Define grid and block sizes  
    dim3 DimGrid(numElements, 1, 1);
    dim3 DimBlock(16, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");


    // Call scan kernel
    scan <<<DimGrid, DimBlock>>> (deviceInput, deviceOutput, numElements);

    //cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    // Copy data from GPU
    cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");
    

	// Partial sums on each block
    hostOutput[0] = 0;
    for (int ii = 1; ii < numOutputElements; ii++) {
         hostOutput[ii] += hostOutput[ii-1];
    }

    wbTime_start(GPU, "Copying input memory to the GPU.");
    // Copy data to GPU  
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostOutput, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice);


    wbTime_start(Compute, "Performing CUDA computation");
    // Call add kernel
    //cudaDeviceSynchronize();

    add<<<DimGrid, DimBlock>>> (deviceInput, deviceOutput, numOutputElements);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    // Copy data from GPU
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");
    wbTime_start(GPU, "Freeing GPU Memory");

    // Free memory from GPU 
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
~      
