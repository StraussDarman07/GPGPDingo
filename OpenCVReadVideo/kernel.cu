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

    const int BLOCK_SIZE = 16;

    // Where does our data start
    int blockStartIndexX = blockIdx.x * blockDim.x - 1;
    int blockStartIndexY = blockIdx.y * blockDim.y - 1;

    // Clamp to edge
    if (blockStartIndexX < 0)
        blockStartIndexX = 0;

    if (blockStartIndexX >= width)
        blockStartIndexX = blockDim.x - 1;

    if (blockStartIndexY < 0)
        blockStartIndexY = 0;

    if (blockStartIndexY >= height)
        blockStartIndexY = blockDim.y - 1;

    // Shared Data
    __shared__ unsigned char pixels[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Where is our data
    unsigned char* cacheInput = input + (blockStartIndexX + blockStartIndexY * width);

    // Linear index (16x16 -> 0..255)
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

    int maxLoadSizeBytes = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2); // 18x18 Block -> 324 Bytes
    int maxIndexBytes = maxLoadSizeBytes / sizeof(short); // 18x18 Block -> Index 162

    if (threadIndex < maxIndexBytes)
    {
        // Calculate offset
        int offsetInBytes = threadIndex * sizeof(short);

        int block_half = (BLOCK_SIZE + 2) / 2;
        int byteRow = offsetInBytes / (BLOCK_SIZE + 2);
        int byteCol = threadIndex % block_half * 2;

        int offset = byteCol + byteRow * width;
        //int offsetBuffer = byteCol + byteRow * (BLOCK_SIZE + 2);

        // Copy Data
        unsigned char* toLoad = cacheInput + offset;
        /**(&pixels[0][0] + offsetBuffer) = *toLoad;
        *(&pixels[0][0] + offsetBuffer + 1) = *(toLoad + 1);*/

        pixels[byteRow][byteCol] = *toLoad;
        pixels[byteRow][byteCol + 1] = *(toLoad + 1);
    }

    __syncthreads();


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
        int indexX = threadIdx.x + 1 + offsetX[i];
        int indexY = threadIdx.y + 1 + offsetY[i];

        unsigned char pixel = pixels[indexY][indexX];
        pointX += pixel * weightsX[i];
        pointY += pixel * weightsY[i];
    }


    // Do Sobel here!
    int index = x + y * width;
    unsigned char * outputData = output + index;
    outputData[0] = sqrtf(pointX * pointX + pointY * pointY);
}


__global__ void sobelTex(unsigned char *output, cudaTextureObject_t *input, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width)
        return;

	unsigned int result = tex2D<unsigned int>(*input,x,y);
		//just that it is white
	unsigned char  test = (unsigned char) (int) result;
    output[y * width + x] = test;
}