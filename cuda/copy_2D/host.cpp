#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <math.h>
#include <numeric> 
#include <vector>
#include <cuda_runtime.h>
#include "extern.h"
#include "kernel.h"

Pixel* hImage;
Pixel* dImage = nullptr;
Pixel* dcopyResult = nullptr;
size_t dImagepitch, hImagepitch;
int  pitch_in_pixels;

// Create CUDA events for timing
cudaEvent_t start, stop;

static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

EXTERN_C void enableDevice() {
	cudaSetDevice(0);
	is_enabled = true;
}

static void set_image_size(const int image_num_blocks_in[NUM_DIMENSIONS]) {
	// image landscape only
	assert(image_num_blocks[vert_dir] <= image_num_blocks[horiz_dir]);

	image_num_blocks[horiz_dir] = image_num_blocks_in[horiz_dir];
	image_num_blocks[vert_dir] = image_num_blocks_in[vert_dir];

	image_num_pixels[vert_dir] = _blocksize * image_num_blocks[vert_dir];
	image_num_pixels[horiz_dir] = _blocksize * image_num_blocks[horiz_dir];
	
	hImagepitch = sizeof(Pixel) * image_num_pixels[horiz_dir];
}

EXTERN_C void initialize(int image_num_blocks_in[NUM_DIMENSIONS])
{
	set_image_size(image_num_blocks_in);
	hImage = (Pixel*)malloc(sizeof(Pixel) * image_num_pixels[horiz_dir] * image_num_pixels[vert_dir]);

	checkCudaError(cudaMallocPitch((void**)&dImage, &dImagepitch, hImagepitch, image_num_pixels[vert_dir]),"cudamalloc pitch failed");
	checkCudaError(cudaMallocPitch((void**)&dcopyResult, &dImagepitch, hImagepitch, image_num_pixels[vert_dir]), "cudamalloc pitch failed");
	pitch_in_pixels = (int)(dImagepitch >> logsizeof_pixel);

	checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
	checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");
	image_init_done = true;
}

EXTERN_C  void MoveImageToDevice(Pixel* data) {
	int width = image_num_pixels[horiz_dir];
	int height = image_num_pixels[vert_dir];
	checkCudaError(cudaMemcpy2DAsync(dImage, dImagepitch, data, hImagepitch, sizeof(Pixel) * width, height, cudaMemcpyHostToDevice), "cuda image to device failed");
}

EXTERN_C void image_copy() {

	// Record the start event
	cudaEventRecord(start, 0);
	dim3 block_dim = dim3(image_num_blocks[horiz_dir], image_num_blocks[vert_dir]);
	dim3 thread_dim_rows = dim3(_blocksize, _blocksize);
	imageCopy << <block_dim, thread_dim_rows >> > (dImage, dcopyResult, pitch_in_pixels);

	// Record the stop event
	cudaEventRecord(stop, 0);

	// Synchronize the stop event to wait for the kernel to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time between the start and stop events
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Print the elapsed time
	std::cout << "Image Copy kernel execution time: " << milliseconds << " ms" << std::endl;
}

EXTERN_C void GetImageFromDevice(Pixel* data) {
	checkCudaError(cudaMemcpy2DAsync(data, hImagepitch, dcopyResult, dImagepitch, sizeof(Pixel) * image_num_pixels[horiz_dir], image_num_pixels[vert_dir], cudaMemcpyDeviceToHost), "cuda image to host failed");
}

EXTERN_C void clear_all() {
	checkCudaError(cudaFree(dImage), "cuda free failed");
	free(hImage); // consider     delete[] hImage;
	image_init_done = false;
}