#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

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

    size_t size  = element_count * sizeof(float);
    CUDA_CHECK(cudaMalloc(ptr, element_count * sizeof(float)));
}

/**
 * Frees memory previously allocated on the GPU.
 * @param ptr Pointer to the device memory to free.
 */
void free_device_memory(float* ptr) {
     if(ptr != nullptr)
        {
  
         CUDA_CHECK(cudaFree(ptr));
            
        }
}


int main() {
    float* d_data = nullptr;
    const size_t element_count = 1024;
    const size_t bytes = element_count * sizeof(float);
    std::vector<float> h_check(element_count, -1.0f);

    std::cout << "Allocating memory for " << element_count << " floats on the GPU..." << std::endl;
    allocate_device_memory(&d_data, element_count);

    // Initialize the device allocation to a known value and verify it from the host.
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    CUDA_CHECK(cudaMemcpy(h_check.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (float value : h_check) {
        if (value != 0.0f) {
            success = false;
            break;
        }
    }

    std::cout << "Allocation and zero-initialization: "
              << (success ? "SUCCESS" : "FAILURE") << std::endl;

    std::cout << "Cleaning up and freeing GPU memory..." << std::endl;
    free_device_memory(d_data);

    return success ? 0 : 1;
}