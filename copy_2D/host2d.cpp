#include "gpu2d.h"
#include <array>
#include <vector>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "cuda.h"

#include <iostream>
#include <algorithm>
#include <numeric> 
#include <chrono>
#include "extern.h"

using namespace std;
const int logblocksize = 4;
const int blocksize = 1 << 4;
# define NUM_DIMENSIONS 2
const int horiz_dir = 0;
const int vert_dir = 1;

size_t dImagepitch;
Pixel* image_host_ptr;
Pixel* dImage = nullptr;
Pixel* dcopyResult = nullptr;
size_t dImagepitch, hImagepitch;
int  pitch_in_pixels;
int  image_num_pixels[NUM_DIMENSIONS];
bool image_init_done = false;
bool is_enabled = false;

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

static void set_image_size(const image_num_blocks[NUM_DIMENSIONS) {
	// image landscape only
	assert(image_num_blocks[vert_dir] <= image_num_blocks[horiz_dir]);
	image_num_pixels[vert_dir] = blocksize * image_num_blocks[vert_dir];
	image_num_pixels[horiz_dir] = blocksize * image_num_blocks[horiz_dir];
	hImagepitch = sizeof(Pixel) * image_num_pixels[horz_dir];
}

EXTERN_C void inititalize(int image_num_blocks[NUM_DIMENSIONS])
{
	set_image_size(constants);
	hImage = (Pixel*)malloc(sizeof(Pixel) * image_num_pixels[horiz_dir] * image_num_pixels[vert_dir]);

	checkCudaErrors(cudaMallocPitch((void**)&dImage, &dImagepitch, hImagepitch, image_num_pixels[vert_dir_]));
	checkCudaErrors(cudaMallocPitch((void**)&dcopyResult, &dImagepitch, hImagepitch, image_num_pixels[vert_dir_]));
	pitch_in_pixels = (int)(dImagepitch >> logsizeof_pixel);

	checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
	checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");
}

EXTERN_C  void MoveImageToDevice(Pixel* data) {
	int width = image_num_pixels[horiz_dir];
	int height = image_num_pixels[vert_dir];
	checkCudaErrors(cudaMemcpy2DAsync(dImage, dImagepitch, data, hImagepitch, sizeof(Pixel) * width, height, cudaMemcpyHostToDevice));
}

EXTERN_C void image_copy() {
	dim3 block_dim = dim3(image_num_blocks[horiz_idx], image_num_blocks[vert_idx]);
	dim3 thread_dim_rows = dim3(blocksize, blocksize);
	imageCopy << <block_dim, thread_dim_rows >> > (dImage, dcopyResult, pitch_in_pixels);
}

EXTERN_C void GetImageFromDevice(Pixel* data, int stream_id) {
	checkCudaErrors(cudaMemcpy2DAsync(data, hImagepitch, dImage, dImagepitch, sizeof(Pixel) * image_num_pixels[horiz_dir], image_num_pixels[vert_dir_], cudaMemcpyDeviceToHost));
}

EXTERN_C void clear_all() {
	checkCudaErrors(cudaFree(dImage));
	free(hImage); // consider     delete[] hImage;
	image_init_done = false;
}