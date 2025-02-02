
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float myTemp = scale * x[idx] + y[idx];
		y[idx] = myTemp;
	}//	Insert GPU SAXPY kernel code here
}

int runGpuSaxpy(int vectorSize) {
	//std::cout << "Hello GPU Saxpy!\n";

    //pinter declarations
	float *xHost, *yHost, *xDevice;
    float* yDevice, *yResult;


    //memory allocation
	xHost = new float[vectorSize * sizeof(float)];   //dynamic mem alloc on host
	yHost = new float[vectorSize * sizeof(float)];   //dynamic mem alloc on host
	yResult = new float[vectorSize * sizeof(float)]; //dynamic mem alloc on host


	// array with random data
	for (int i = 0; i < vectorSize; i++) {
		xHost[i] = (float)(rand() % 1000);  // randmom, modulo val doesnt matter
		yHost[i] = (float)(rand() % 1000); //random, modulo val doesnt matter
	}
	float scale = 11; // random, could have used rand()

    //mem allocation on gpu
	cudaMalloc((void **)&xDevice, vectorSize * sizeof(float)); //mem allocation on gpu
	cudaMalloc((void **)&yDevice, vectorSize * sizeof(float));  //mem allocation on gpu

    //cpu to gpu mem transfer
	cudaMemcpy(xDevice, xHost, vectorSize * sizeof(float), cudaMemcpyHostToDevice); //copyinng memory from cpu to gpu
	cudaMemcpy(yDevice, yHost, vectorSize * sizeof(float), cudaMemcpyHostToDevice); //copying memory from cpu to gpu

	// Call saxypy kernel
	saxpy_gpu<<<std::ceil(vectorSize / 256), 256>>>(xDevice, yDevice, scale, vectorSize);

    //gpu to cpu mem transfer
	cudaMemcpy(yResult, yDevice, vectorSize * sizeof(float), cudaMemcpyDeviceToHost); // copyig results to cpu from gpu

    //errors
	int errCount = verifyVector(xHost, yHost, yResult, scale, vectorSize); // number of errors
	std::cout << "Found " << errCount << " / " << vectorSize << " errors \n";

    //free memory from cpu
	delete xHost;
	delete yHost;
	delete yResult;

    //free mem from gpu
	cudaFree(xDevice);
	cudaFree(yDevice);

	//	Insert code here
	//std::cout << "Lazy, you are!\n";
	//std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//x and y on gpu
	float xDevice, yDevice;

	//declaration of states
	curandState_t state;

	//initialization
	curand_init(clock64(), t_idx, 0, &state);

	if (t_idx < pSumSize) {
		for (int i = 0; i < sampleSize; i++) {

			xDevice = curand_uniform(&state); yDevice = curand_uniform(&state);

			// dist calc
			float distance = xDevice * xDevice  +  yDevice * yDevice;  // dist = x^2 +
			if (distance <= 1)
				pSums[t_idx]++;
		}
	}//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
		int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	int startIdx = t_idx * reduceSize;

	int endIdx = (t_idx + 1) * reduceSize;


	if (endIdx <= pSumSize && t_idx < (pSumSize / reduceSize)) {

		for (int i = startIdx; i < endIdx; i++) {

			totals[t_idx] += pSums[i];

		}
}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

 double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, uint64_t reduceThreadCount, uint64_t reduceSize) {
 //declarations
    double approxPi = 0; uint64_t sum_total = 0;
    
    uint64_t *p_Sums, *totals_Device, *totals_Host;
    
    uint64_t p_Sum_Size = generateThreadCount;
    
    
    //Allocating on Device
    cudaMalloc((uint64_t**)&p_Sums, p_Sum_Size * sizeof(uint64_t));
    
    cudaMalloc((uint64_t**)&totals_Device, reduceThreadCount * sizeof(uint64_t));
    
    
    //Allocation on host
    totals_Host = new uint64_t[reduceThreadCount];
	//Calling kernel
    
    generatePoints<<<(generateThreadCount/256) + 1, 256>>>(p_Sums, p_Sum_Size, sampleSize);
    
    reduceCounts<<<(generateThreadCount/(reduceThreadCount * reduceSize)) + 1, reduceThreadCount>>>(p_Sums, totals_Device, generateThreadCount, reduceSize);
    
    //memTransfer Dev to Host
    cudaMemcpy(totals_Host, totals_Device, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    //iteration for approximate val
    for (int i = 0; i < reduceThreadCount; i++)
        sum_total += totals_Host[i];
    approxPi = double(sum_total * 4) / double(generateThreadCount * sampleSize);
    
    //std :: cout << "Sneaky, you are..\n";
    //std :: cout << "Compute pi, upi must!\n";
    return approxPi;
}
	
