#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void toOneChannel(unsigned char *data, int width, int height, int components)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || column >= width)
        return;

    unsigned char * threadData = data + (components * (column + row * width));

    for (int i = 0; i < components - 1; i++)
    {
        threadData[i] = 0;
    }
}

__global__ void toGrayScale(unsigned char *data, int width, int height, int components)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || column >= width)
        return;

    unsigned char * threadData = data + (components * (column + row * width));

    float partRed = 0.299f;
    float partGreen = 0.587f;
    float partBlue = 0.114;

    unsigned char greyScale = partBlue * threadData[0] + partGreen * threadData[1] + partRed * threadData[2];

    threadData[0] = greyScale;
    threadData[1] = greyScale;
    threadData[2] = greyScale;

}