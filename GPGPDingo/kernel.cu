
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"

#include <stdio.h>
#include <cstdlib>
#include <math.h>
// Matrix is ROW Major in memory!

const int matWidth = 1024;
const int size = matWidth * matWidth;

void matMulHost(float* result, const float* a, const float* b, int size)
{
	for (int row = 0; row < size; row++)
	{
		for (int column = 0; column < size; column++)
		{
			result[column + row * size] = 0;
			for (int index = 0; index < size; index++)
			{
				result[column + row * size] += a[row * size + index] * b[column + size * index];
			}
		}


	}
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* wA is A's width and wB is B's width
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__global__ void matMulCuda(float* result, const float* a, const float* b, int matWidth)
{
	int row = blockIdx.x;
	int column = blockIdx.y;

	int resultIndex = column + row * matWidth;
	if (resultIndex < matWidth*matWidth)
	{
		result[resultIndex] = 0;
		for (int index = 0; index < matWidth; index++)
		{
			result[resultIndex] += a[row * matWidth + index] * b[column + matWidth * index];
		}
	}
}

float getRandom()
{
	return ((float)rand()) / ((float)RAND_MAX);
}

#define CUDA_ERROR_CHECK(Value) if (Value != cudaSuccess) {printf("cudafailed!"); return 1;};

int main()
{
	srand((unsigned)139213);
	// Init Random Seed
	/*std::default_random_engine randomEngine;
	std::uniform_real_distribution<float> distribution(0.f, 10.f);
	auto dice = std::bind(distribution, randomEngine);*/
	float *aCpu = (float*)malloc(sizeof(float) * size);
	for (int i = 0; i < size; ++i)
		*(aCpu + i) = getRandom();

	float *bCpu = (float*)malloc(sizeof(float) * size);
	for (int i = 0; i < size; ++i)
		*(bCpu + i) = getRandom();

	float *cpuResult = (float*)malloc(sizeof(float) * size);
	float *hostCudaResult = (float*)malloc(sizeof(float) * size);

	// Create timer
	StopWatchInterface *t;
	if (!sdkCreateTimer(&t)) {
		printf("timercreate failed\n");
		exit(-1);
	}

	sdkResetTimer(&t);
	sdkStartTimer(&t);
	matMulHost(cpuResult, aCpu, bCpu, matWidth);
	sdkStopTimer(&t);

	printf("Zeitdauer (CPU): %f\n", sdkGetTimerValue(&t));

	// Resource alloc
	dim3 gridDim;
	gridDim.x = matWidth;
	gridDim.y = matWidth;
	float *cudaA, *cudaB, *cudaResult;
	CUDA_ERROR_CHECK(cudaMalloc(&cudaA, sizeof(float) * size));
	CUDA_ERROR_CHECK(cudaMalloc(&cudaB, sizeof(float) * size));
	CUDA_ERROR_CHECK(cudaMalloc(&cudaResult, sizeof(float) * size));

	cudaMemset(cudaResult, 0, sizeof(float) * size);

	// Warmup
	sdkResetTimer(&t);
	sdkStartTimer(&t);
	CUDA_ERROR_CHECK(cudaMemcpy(cudaA, aCpu, 2 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(cudaB, bCpu, 2 * sizeof(float), cudaMemcpyHostToDevice));
	matMulCuda << <gridDim, 1 >> > (cudaResult, cudaA, cudaB, 2);
	cudaDeviceSynchronize();
	cudaMemcpy(hostCudaResult, cudaResult, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	sdkStopTimer(&t);
	printf("Zeitdauer (GPU-Warmup): %f\n", sdkGetTimerValue(&t));

	// Measurement
	sdkResetTimer(&t);
	sdkStartTimer(&t);
	CUDA_ERROR_CHECK(cudaMemcpy(cudaA, aCpu, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(cudaB, bCpu, size * sizeof(float), cudaMemcpyHostToDevice));
	matMulCuda << <gridDim, 1 >> > (cudaResult, cudaA, cudaB, matWidth);
	cudaDeviceSynchronize();
	cudaMemcpy(hostCudaResult, cudaResult, size * sizeof(float), cudaMemcpyDeviceToHost);
	sdkStopTimer(&t);
	printf("Zeitdauer (GPU): %f\n", sdkGetTimerValue(&t));

	dim3 threads(32, 32);
	dim3 grid(matWidth / threads.x, matWidth / threads.y);

	// Cuda Impl Measurement
	sdkResetTimer(&t);
	sdkStartTimer(&t);
	CUDA_ERROR_CHECK(cudaMemcpy(cudaA, aCpu, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(cudaB, bCpu, size * sizeof(float), cudaMemcpyHostToDevice));
	matrixMulCUDA<32> << < grid, threads >> >(cudaResult, cudaA, cudaB, matWidth, matWidth);
	cudaDeviceSynchronize();
	cudaMemcpy(hostCudaResult, cudaResult, size * sizeof(float), cudaMemcpyDeviceToHost);
	sdkStopTimer(&t);
	printf("Zeitdauer (GPU(CudaImpl)): %f\n", sdkGetTimerValue(&t));

	float absError = 0.f;
	for (int i = 0; i < size; ++i)
	{
		absError += fabs(cpuResult[i] - hostCudaResult[i]);
	}

	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaResult);
	
	printf("Absolute Error: %f\n", absError);
	free(aCpu);
	free(bCpu);
	free(cpuResult);
	free(hostCudaResult);

	sdkDeleteTimer(&t);
	getchar();
	return 0;
}





