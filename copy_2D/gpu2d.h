#ifndef GPU2D_H
#define GPU2D_H

// Include the CUDA runtime header
#include <cuda_runtime.h>

typedef unsigned char Pixel;  //  the pixel data type, image input and display output
const  int logsizeof_pixel = 1;

__global__ void imageCopy(const Pixel* image, Pixel* result, const int width);

#endif // GPU2D_H