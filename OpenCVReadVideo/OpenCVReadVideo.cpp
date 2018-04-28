// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

using namespace cv;
using namespace std;

extern void toOneChannel(unsigned char *data, int width, int height, int components);

void singleChannelCuda(Mat& output)
{
	void* deviceMem;
	int elemSize = output.elemSize();
	size_t size_in_bytes = elemSize * output.cols * output.rows;
	cudaMalloc(&deviceMem, size_in_bytes);

	cudaMemcpy(deviceMem, output.data, size_in_bytes, cudaMemcpyHostToDevice);

	void *args[] = { &deviceMem , &output.cols, &output.rows, &elemSize };
	cudaLaunchKernel<void>(&toOneChannel, dim3(output.cols / 16 + 1, output.rows / 16 + 1), dim3(16, 16), args);

	cudaDeviceSynchronize();
	cudaMemcpy(output.data, deviceMem, size_in_bytes, cudaMemcpyDeviceToHost);

	cudaFree(deviceMem);
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

	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		Mat output;
		cap >> frame; // get a new frame from camera

		if (frame.dims == 0) { // we're done
			break;
		}

		output = frame.clone();
		singleChannelCuda(output);

		imshow("edges", output);
		if (waitKey(1) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;



	//cvtColor(frame, edges, COLOR_BGR2GRAY);
	//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	//		Sobel(frame, edges, frame.depth(), 2, 2);
	//		Canny(edges, edges, 0, 30, 3);
	//		imshow("edges", edges);
}
