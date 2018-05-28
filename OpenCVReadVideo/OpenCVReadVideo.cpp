// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "helper_functions.h"

using namespace cv;
using namespace std;

extern void toOneChannel(unsigned char *data, int width, int height, int components);
extern void toGrayScale(unsigned char *output, unsigned char *input, int width, int height, int components);
extern void sobel(unsigned char *output, unsigned char *input, int width, int height);
extern void sobelTex(unsigned char *output, cudaTextureObject_t input, int width, int height);

void singleChannelCuda(Mat& output)
{
	void* deviceMem;
	size_t elemSize = output.elemSize();
	size_t size_in_bytes = elemSize * output.cols * output.rows;
	cudaMalloc(&deviceMem, size_in_bytes);

	cudaMemcpy(deviceMem, output.data, size_in_bytes, cudaMemcpyHostToDevice);

	void *args[] = { &deviceMem , &output.cols, &output.rows, &elemSize };
	cudaLaunchKernel<void>(&toOneChannel, dim3(output.cols / 16 + 1, output.rows / 16 + 1), dim3(16, 16), args);

	cudaDeviceSynchronize();
	cudaMemcpy(output.data, deviceMem, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaFree(deviceMem);
}

void greyScaleCuda(Mat& frame, void* output)
{
	size_t elemSize = frame.elemSize();
	size_t size_in_bytes = elemSize * frame.cols * frame.rows;

	void* input;
	cudaMalloc(&input, size_in_bytes);
	cudaMemcpy(input, frame.data, size_in_bytes, cudaMemcpyHostToDevice);

	void *args[] = { &output, &input , &frame.cols, &frame.rows, &elemSize };
	cudaLaunchKernel<void>(&toGrayScale, dim3(frame.cols / 16 + 1, frame.rows / 16 + 1), dim3(16, 16), args);

	cudaDeviceSynchronize();
	cudaFree(input);
}

void sobelCuda(Mat& outputFrame, void* input)
{
	size_t size_in_bytes = outputFrame.cols * outputFrame.rows;

	void* output;
	cudaMalloc(&output, size_in_bytes);

	void *args[] = { &output, &input , &outputFrame.cols, &outputFrame.rows };
	cudaError_t error = cudaLaunchKernel<void>(&sobel, dim3(outputFrame.cols / 16 + 1, outputFrame.rows / 16 + 1), dim3(16, 16), args);
	cudaDeviceSynchronize();
	cudaMemcpy(outputFrame.data, output, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaFree(output);
}

void sobelCudaTex(Mat& outputFrame, void* input)
{
	size_t size_in_bytes = outputFrame.cols * outputFrame.rows;


	//we have already a greyscaled picture and I'm not sure about the channels anymore
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, outputFrame.cols, outputFrame.rows);
	cudaMemcpyToArray(cuArray, 0, 0, input, size_in_bytes, cudaMemcpyDeviceToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;// cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	void* output;
	cudaMalloc(&output, size_in_bytes);

	void *args[] = { &output, &texObj , &outputFrame.cols, &outputFrame.rows };
	cudaError_t error = cudaLaunchKernel<void>(&sobelTex, dim3(outputFrame.cols / 16 + 1, outputFrame.rows / 16 + 1), dim3(16, 16), args);

	cudaDeviceSynchronize();

	cudaMemcpy(outputFrame.data, output, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaDestroyTextureObject(texObj);

	cudaFreeArray(cuArray);
	cudaFree(output);
}

int main(int, char**)
{
	//	VideoCapture cap("Z:/Videos/robotica_1080.mp4"); // open the default camera
	VideoCapture cap("..\\Videos\\robotica_1080.mp4");
	//	VideoCapture cap("C:/Users/fischer/Downloads/Bennu4k169Letterbox_h264.avi"); // open the default camera
	//	VideoCapture cap("D:/Users/fischer/Videos/fireworks.mp4");
	//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.mp4");
	//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.avi");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	bool firstFrame = true;
	void* greyScaleBuffer = nullptr;
	size_t frameBufferSize = 0;

	StopWatchInterface *t;
	if (!sdkCreateTimer(&t)) {
		printf("timercreate failed\n");
		exit(-1);
	}

	float time = 0;
	int frameCount = 0;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera

		if (frame.dims == 0) { // we're done
			break;
		}

		if (firstFrame)
		{
			frameBufferSize = frame.cols * frame.rows;
			cudaMalloc(&greyScaleBuffer, frameBufferSize);
			firstFrame = false;
		}

		greyScaleCuda(frame, greyScaleBuffer);


		Mat output(frame.rows, frame.cols, CV_8UC1);
		// Create timer

		sdkStartTimer(&t);
		sobelCudaTex(output, greyScaleBuffer);
		sdkStopTimer(&t);

		time += sdkGetTimerValue(&t);
		frameCount++;
		sdkResetTimer(&t);
		imshow("edges", output);
		if (waitKey(1) >= 0) break;
	}

	cudaFree(greyScaleBuffer);
	printf("Average Time per Frame: %fms", time / frameCount);
	getchar();
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;



	//cvtColor(frame, edges, COLOR_BGR2GRAY);
	//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	//		Sobel(frame, edges, frame.depth(), 2, 2);
	//		Canny(edges, edges, 0, 30, 3);
	//		imshow("edges", edges);
}
