#include "kernel.h"
#include <cuda_runtime.h>
#include <algorithm>

// Kernel to copy data from src to dst using rind
__global__ void copyWithRandomIndices(int* src, int* dst, int* rind, int size, int items_per_thread) {
    // Calculate the global thread ID
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = thread_idx * items_per_thread;
	int idx_end = min(idx + items_per_thread, size);
    // Ensure we don't access out of bounds
	for (; idx < idx_end; idx++) {
		int randIdx = rind[idx];
		dst[randIdx] = src[idx];
	}
}

// Copy data from src to dst, with shared memory, and dummy array
__global__ void copyWithRandomIndicesShared(int* src, int* dst, int* rind, int size, int items_per_thread) {
    // Allocate shared memory
    extern __shared__ int sharedSrc[];

    // Calculate the global thread ID
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;


    // Load data from global memory to shared memory
    int idx0 = globalIdx * items_per_thread;
    int idx_end = min(idx0 + items_per_thread, size);
    // Ensure we don't access out of bounds
    for (int idx=idx0; idx < idx_end; idx++) {
        sharedSrc[idx-idx0] = src[idx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Ensure we don't access out of bounds
    for (int idx = idx0; idx < idx_end; idx++) {
        int randIdx = rind[idx];
        dst[randIdx] = sharedSrc[idx-idx0];
    }
}


// Copy data from src to dst, with shared memory, and dummy array
__global__ void copyShared(int* src, int* dst, int* dummy, int size) {
    // Allocate shared memory
    extern __shared__ int sharedSrc[];

    // Calculate the global thread ID
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the local thread ID within the block
    int localIdx = threadIdx.x;

    // Load data from global memory to shared memory
    if (globalIdx < size) {
        sharedSrc[localIdx] = src[globalIdx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Ensure we don't access out of bounds
    if (globalIdx < size) {
        dst[globalIdx] = sharedSrc[localIdx];
    }
}

// Copy data from src to dst using rind
__global__ void copy_(int* src, int* dst, int* dummy, int size) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access out of bounds
    if (idx < size) {
        dst[idx] = src[idx];
    }
}