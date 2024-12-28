#ifndef KERNEL_H
#define KERNEL_H

// Include the CUDA runtime header
#include <cuda_runtime.h>

// Kernel to copy data from src to dst using rind
__global__ void copyWithRandomIndices(int* src, int* dst, int* rind, int size, int items_per_thread);

// Copy data from src to dst, with shared memory, with random indices
__global__ void copyWithRandomIndicesShared(int* src, int* dst, int* rind, int size, int items_per_thread);

// Copy data from src to dst, with shared memory, and dummy array read
__global__ void copyShared(int* src, int* dst, int* dummy, int size);
// Copy data from src to dst using rind, and dummy array read
__global__ void copy_(int* src, int* dst, int* dummy, int size);

#endif // KERNEL_H