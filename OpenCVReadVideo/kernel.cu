#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void addKernel(unsigned char *data, int width, int height, int components)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= height || column >= width)
		return;

	unsigned char * threadData = data + (components * (column + row * width));

	for (int i = 0; i < components-1; i++)
	{
		threadData[i] = 0; 
	}
}