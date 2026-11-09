#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for robust CUDA error checking
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)               \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

/**
 * Safely allocates memory on the GPU (Device).
 * @param ptr Pointer to the device pointer to be allocated.
 * @param element_count Number of float elements to allocate.
 */
void allocate_device_memory(float** ptr, size_t element_count) {
    CUDA_CHECK(cudaMalloc(ptr, element_count * sizeof(float)));
}

/**
 * Frees memory previously allocated on the GPU.
 * @param ptr Pointer to the device memory to free.
 */
void free_device_memory(float* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

int main() {
    // Array of device pointers to manage multiple GPU allocations
    float* d_arrays[3] = {nullptr, nullptr, nullptr};

    // Element count per allocation
    const size_t element_counts[3] = {512, 1024, 2048};
    
    // Total allocation size in bytes for each array
    size_t bytes[3];
    for (size_t i = 0; i < std::size(element_counts); ++i) {
        bytes[i] = element_counts[i] * sizeof(float);
    }

    // Step 1: Allocate memory on the GPU for each array
    std::cout << "Allocating memory for three device arrays..." << std::endl;
    for (size_t i = 0; i < std::size(element_counts); ++i) {
        allocate_device_memory(&d_arrays[i], element_counts[i]);
    }

    // Step 2: Initialize device arrays with unique byte patterns
    std::cout << "Initializing the device arrays..." << std::endl;
    for (size_t i = 0; i < std::size(element_counts); ++i) {
        CUDA_CHECK(cudaMemset(d_arrays[i], static_cast<int>(i), bytes[i]));
    }

    // Step 3: Release all device memory back to VRAM
    std::cout << "Cleaning up and freeing GPU memory..." << std::endl;
    for (size_t i = 0; i < std::size(element_counts); ++i) {
        free_device_memory(d_arrays[i]);
    }

    return 0;
}