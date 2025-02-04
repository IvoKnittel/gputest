#ifndef GPU2D_H
#define GPU2D_H

# define NUM_DIMENSIONS 2
# define _blocksize 16

// Include the CUDA runtime header
#include <cuda_runtime.h>

typedef unsigned char Pixel;  //  the pixel data type, image input and display output
const  int logsizeof_pixel = 0;
bool image_init_done = false;
bool is_enabled = false;

__global__ void imageCopy(const Pixel* image, Pixel* result, const int width);

#endif // GPU2D_H