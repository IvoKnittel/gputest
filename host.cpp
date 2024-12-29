#include <iostream>
#include <algorithm>
#include <numeric> 
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "kernel.h"
#include "extern.h"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

EXTERN_C float test_copy_allkinds(const int blockSize, const int numElements, const bool use_shared, const bool is_consecutive) {
	/*  Data in consecutive order on the GPU device is copied without shared memory
	blocksize = number of threads
    number of elements to be copied */ 

    // Allocate host memory
    int* h_src = new int[numElements];
    int* h_dst = new int[numElements];

    // Initialize source array
    for (int i = 0; i < numElements; ++i) {
        h_src[i] = i;
    }

    // Random permutation of indices
	// this is needed only for fair comparison
    std::vector<int> h_rind(numElements);
    std::iota(h_rind.begin(), h_rind.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(h_rind.begin(), h_rind.end(), g);

    // Allocate device memory
    int* d_src;
    int* d_dst;
    int* d_rind;
    checkCudaError(cudaMalloc((void**)&d_src, numElements * sizeof(int)), "cudaMalloc d_src failed");
    checkCudaError(cudaMalloc((void**)&d_dst, numElements * sizeof(int)), "cudaMalloc d_dst failed");
    checkCudaError(cudaMalloc((void**)&d_rind, numElements * sizeof(int)), "cudaMalloc d_rind failed");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_src, h_src, numElements * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_src failed");
    checkCudaError(cudaMemcpy(d_rind, h_rind.data(), numElements * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_rind failed");

    // Calculate the number of blocks needed
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    // Measure time for the copy with shared memory
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
	if (use_shared && is_consecutive) {
		copyShared << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_src, d_dst, d_rind, numElements);
	}
	else if (use_shared && !is_consecutive) {
		copyWithRandomIndicesShared << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_src, d_dst, d_rind, numElements, 1);
	}
	else if (!use_shared && is_consecutive) {
		copy_ << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_src, d_dst, d_rind, numElements);
	}
	else {
		copyWithRandomIndices << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_src, d_dst, d_rind, numElements, 1);
	}
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime failed");

    // Copy data from device to host
    checkCudaError(cudaMemcpy(h_dst, d_dst, numElements * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy h_dst failed");

    // Verify the result
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_dst[i] != h_src[i]) {
            std::cerr << "Ref: Mismatch at index " << i << " : " << h_dst[i] << " != " << h_src[i] << std::endl;
            success = false;
        }
    }
    if (success) {
        std::cout << "Copy successful!" << std::endl;
    } else {
		milliseconds = -1.0;
    }

    // Print the timing result
    std::cout << "Time for copy: " << milliseconds << " ms" << std::endl;

    // Clean up
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_rind);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

EXTERN_C float test_random_index_copy(const int blockSize, const int numElements)
{
	const int items_per_thread = 1;
    // Allocate host memory
    int* h_src = new int[numElements];
    int* h_dst = new int[numElements];
    int* h_dst_ref = new int[numElements];
    std::vector<int> h_rind(numElements);

    // Initialize source array
    for (int i = 0; i < numElements; ++i) {
        h_src[i] = i;
    }

    // Generate random permutation of indices
    std::iota(h_rind.begin(), h_rind.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(h_rind.begin(), h_rind.end(), g);

    // Allocate device memory
    int* d_src;
    int* d_dst;
    int* d_rind;
    checkCudaError(cudaMalloc((void**)&d_src, numElements * sizeof(int)), "cudaMalloc d_src failed");
    checkCudaError(cudaMalloc((void**)&d_dst, numElements * sizeof(int)), "cudaMalloc d_dst failed");
    checkCudaError(cudaMalloc((void**)&d_rind, numElements * sizeof(int)), "cudaMalloc d_rind failed");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_src, h_src, numElements * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_src failed");
    checkCudaError(cudaMemcpy(d_rind, h_rind.data(), numElements * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_rind failed");

    // Calculate the number of blocks needed
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    // Measure time for the random index copy with shared memory
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
    copyWithRandomIndices << <numBlocks, blockSize, blockSize * sizeof(int) >> > (d_src, d_dst, d_rind, numElements, items_per_thread);
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime failed");

    // Copy data from device to host
    checkCudaError(cudaMemcpy(h_dst, d_dst, numElements * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy h_dst failed");

    bool success = true;
    // Verify the result
    for (int i = 0; i < numElements; ++i) {
        int r_ind = h_rind[i];
        if (h_dst[r_ind] != h_src[i]) {
            std::cerr << "Mismatch at index " << r_ind << " : " << h_dst[r_ind] << " != " << h_src[i] << std::endl;
            success = false;
        }
    }
    if (success) {
        std::cout << "Random index copy successful!" << std::endl;
    }
    else 
    {
		milliseconds = -1.0;
    }

    // Print the timing result
    std::cout << "Time for random index copy: " << milliseconds << " ms" << std::endl;

    // Clean up
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_rind);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return milliseconds;
}