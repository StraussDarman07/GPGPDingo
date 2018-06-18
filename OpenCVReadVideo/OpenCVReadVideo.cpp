#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "helper_functions.h"

using namespace cv;
using namespace std;

extern void toOneChannel(unsigned char *data, int width, int height, int components);
extern void toGrayScale(unsigned char *output, unsigned char *input, int width, int height, int components);
extern void sobel(unsigned char *output, unsigned char *input, int width, int height);

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
	cudaLaunchKernel<void>(&sobel, dim3(outputFrame.cols / 16 + 1, outputFrame.rows / 16 + 1), dim3(16, 16), args);

	cudaDeviceSynchronize();
	cudaMemcpy(outputFrame.data, output, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaFree(output);
}

struct Stream
{
	cudaStream_t cudaStream;
	Mat input_frame;
	Mat output_frame;
	void* d_inout_buffer;
	void* d_buffer;
};

void grayScaleSobelStream(Stream* stream)
{
	// Copy Input frame to GPU
	size_t size_in_bytes = stream->input_frame.cols * stream->input_frame.rows;
	size_t elemSize = stream->input_frame.elemSize();
	cudaMemcpyAsync(stream->d_inout_buffer, stream->input_frame.data, size_in_bytes * elemSize, cudaMemcpyHostToDevice, stream->cudaStream);

	// Greyscale
	void *argsGreyScale[] = { &stream->d_buffer, &stream->d_inout_buffer , &stream->input_frame.cols, &stream->input_frame.rows, &elemSize };
	cudaLaunchKernel<void>(&toGrayScale, dim3(stream->input_frame.cols / 16 + 1, stream->input_frame.rows / 16 + 1), dim3(16, 16), argsGreyScale, 0, stream->cudaStream);

	// Sobel
	void *argsSobel[] = { &stream->d_inout_buffer, &stream->d_buffer , &stream->input_frame.cols, &stream->input_frame.rows };
	cudaLaunchKernel<void>(&sobel, dim3(stream->input_frame.cols / 16 + 1, stream->input_frame.rows / 16 + 1), dim3(16, 16), argsSobel, 0, stream->cudaStream);

	// Copy output frame to cpu
	stream->output_frame = Mat(stream->input_frame.rows, stream->input_frame.cols, CV_8UC1);
	cudaMemcpyAsync(stream->output_frame.data, stream->d_inout_buffer, size_in_bytes, cudaMemcpyDeviceToHost, stream->cudaStream);
}

int main(int, char**)
{
	VideoCapture cap("..\\Videos\\robotica_1080.mp4");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	int frameCount = 0;
	float global_time = 0;
	StopWatchInterface *t;
	if (!sdkCreateTimer(&t)) {
		printf("timercreate failed\n");
		exit(-1);
	}

	namedWindow("edges", 1);

	Stream stream1, stream2;
	cudaStreamCreate(&stream1.cudaStream);
	cudaStreamCreate(&stream2.cudaStream);

	cap >> stream1.input_frame; // get a new frame from camera
	cap >> stream2.input_frame;

	size_t frameBufferSize = stream1.input_frame.cols * stream1.input_frame.rows;
	cudaMalloc(&stream1.d_inout_buffer, frameBufferSize * stream1.input_frame.elemSize());
	cudaMalloc(&stream2.d_inout_buffer, frameBufferSize * stream2.input_frame.elemSize());
	cudaMalloc(&stream1.d_buffer, frameBufferSize);
	cudaMalloc(&stream2.d_buffer, frameBufferSize);


	grayScaleSobelStream(&stream1);
	auto currentStream = &stream2;

	sdkStartTimer(&t);
	for (;;)
	{
		if (currentStream->input_frame.dims == 0)
			break;
		frameCount++;
#if 1
		grayScaleSobelStream(currentStream);
		if (waitKey(1) >= 0) break;

		if (currentStream == &stream2)
			currentStream = &stream1;
		else
			currentStream = &stream2;

		// Wait for currentStream
		cudaStreamSynchronize(currentStream->cudaStream);
		imshow("edges", currentStream->output_frame);
		cap >> currentStream->input_frame;

#else
		frameCount++;
		sdkStartTimer(&t);

		greyScaleCuda(frame, greyScaleBuffer);

		Mat output(frame.rows, frame.cols, CV_8UC1);
		sobelCuda(output, greyScaleBuffer);
		imshow("edges", output);
		sdkStopTimer(&t);
		global_time += sdkGetTimerValue(&t);
		sdkResetTimer(&t);
		if (waitKey(1) >= 0) break;
#endif
	}

	
	if (currentStream == &stream2)
		currentStream = &stream1;
	else
		currentStream = &stream2;

	cudaStreamSynchronize(currentStream->cudaStream);
	imshow("edges", currentStream->output_frame);
	
	sdkStopTimer(&t);
	global_time += sdkGetTimerValue(&t);


	cudaFree(stream1.d_inout_buffer);
	cudaFree(stream2.d_inout_buffer);
	cudaFree(stream1.d_buffer);
	cudaFree(stream2.d_buffer);
	printf("Average Time per Frame(global): %fms\n", global_time / frameCount);
	getchar();
	return 0;
	}