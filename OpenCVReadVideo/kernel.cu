#include "cuda_runtime.h"
#include "math_functions.h"
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

__global__ void toGrayScale(unsigned char *output, unsigned char *input, int width, int height, int components)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || column >= width)
        return;

    int index = column + row * width;
    unsigned char * threadData = input + components * index;
    unsigned char * outputData = output + index;

    const float partRed = 0.299f;
    const float partGreen = 0.587f;
    const float partBlue = 0.114;

    unsigned char greyScale = partBlue * threadData[0] + partGreen * threadData[1] + partRed * threadData[2];

    outputData[0] = greyScale;
}

__global__ void sobel(unsigned char *output, unsigned char *input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width)
        return;

    // Sobel weights
    float weightsX[9] = { -1, -2, -1,
                           0,  0,  0,
                           1,  2,  1 };

    float weightsY[9] = { -1,  0,  1,
                          -2,  0,  2,
                          -1,  0,  1 };

    int offsetY[9] = { -1,  -1,  -1,
                        0,   0,   0,
                        1,   1,   1 };

    int offsetX[9] = { -1,   0,   1,
                       -1,   0,   1,
                       -1,   0,   1 };


    float pointX = 0.f;
    float pointY = 0.f;
#pragma unroll
    for (int i = 0; i < 9; i++)
    {
        int index = (x + offsetX[i]) + (y + offsetY[i]) * width;

        unsigned char pixel = *(input + index);
        pointX += pixel * weightsX[i];
        pointY += pixel * weightsY[i];
    }


    // Do Sobel here!
    int index = x + y * width;
    unsigned char * outputData = output + index;
    outputData[0] = sqrtf(pointX * pointX + pointY * pointY);
}