#include "gpu2d.h"

__global__ void imageCopy(const BigPixel* image, BigPixel* result, const int width) {

	int cBegin = width * blocksize * blockIdx.y + blocksize * blockIdx.x;

	// Declaration of the shared copyTo of C block
	__shared__ BigPixel Cs[blocksize][blocksize];
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	Cs[ty][tx] = image[cBegin + width * ty + tx];

	__syncthreads();

	result[cBegin + width * ty + tx] = Cs[ty][tx];

	__syncthreads();
}

