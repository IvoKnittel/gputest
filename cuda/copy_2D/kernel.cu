#include "kernel.h"

__global__ void imageCopy(const Pixel* image, Pixel* result, const int width) {

	int cBegin = width * _blocksize * blockIdx.y + _blocksize * blockIdx.x;

	// Declaration of the shared copyTo of C block
	__shared__ Pixel Cs[_blocksize][_blocksize];
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	Cs[ty][tx] = image[cBegin + width * ty + tx];

	__syncthreads();

	result[cBegin + width * ty + tx] = Cs[ty][tx];

	__syncthreads();
}

