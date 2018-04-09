
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void matMulCuda(float* result, const float* a, const float* b, int size)
{
	int row = blockIdx.x;
	int column = blockIdx.y;

	result[column + row * size] = 0;
	for (int index = 0; index < size; index++)
	{
		result[column + row * size] += a[row * size + index] * b[column + size * index];
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
	matMulHost(cpuResult, aCpu, bCpu, matWidth);


	float *cudaA, *cudaB, *cudaResult;
	CUDA_ERROR_CHECK(cudaMalloc(&cudaA, sizeof(float) * size));
	CUDA_ERROR_CHECK(cudaMalloc(&cudaB, sizeof(float) * size));
	CUDA_ERROR_CHECK(cudaMalloc(&cudaResult, sizeof(float) * size));

	CUDA_ERROR_CHECK(cudaMemcpy(cudaA, aCpu, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(cudaB, bCpu, size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim;
	blockDim.x = 1024;
	blockDim.y = 1024;

	matMulCuda << <blockDim, 1 >> >(cudaResult, cudaA, cudaB, matWidth);
	cudaDeviceSynchronize();

	cudaMemcpy(hostCudaResult, cudaResult, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(cudaA);
	cudaFree(cudaB);
	cudaFree(cudaResult);

	for (int i = 0; i < size; ++i)
	{
		if (cpuResult[i] != hostCudaResult[i])
			printf("SHITY");
	}

	free(aCpu);
	free(bCpu);
	free(cpuResult);
	free(hostCudaResult);
	return 0;
}





