#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to wrap CUDA runtime calls and exit on error with file & line context
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)               \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// Allocates linear GPU device memory (in bytes) for a given float count
void allocate_device_memory(float** ptr, size_t element_count) {
    CUDA_CHECK(cudaMalloc(ptr, element_count * sizeof(float)));
}

// Safely releases GPU memory if the pointer is non-null
void free_device_memory(float* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Copies an array of floats from Host (CPU) to Device (GPU)
void transfer_to_device(float* dest, const float* src, size_t element_count) {
    size_t size = element_count * sizeof(float);
    CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

// Copies an array of floats from Device (GPU) back to Host (CPU)
void transfer_to_host(float* dest, const float* src, size_t element_count) {
    size_t size = element_count * sizeof(float);
    CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}

int main() {
    const size_t N = 8;
    std::vector<float> h_input = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f};
    std::vector<float> h_output(N, 0.0f);
    float* d_data = nullptr;

    // 1. Allocate device memory
    allocate_device_memory(&d_data, N);

    // 2. Perform round-trip transfer: Host -> Device -> Host
    transfer_to_device(d_data, h_input.data(), N);
    transfer_to_host(h_output.data(), d_data, N);

    // 3. Verify exact bitwise match (safe because no arithmetic occurred)
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i]) {
            success = false;
            break;
        }
    }

    std::cout << "Verified Transfer: " << (success ? "SUCCESS" : "FAILURE") << std::endl;

    std::cout << "Returned Data: ";
    for (float val : h_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 4. Free device resources
    free_device_memory(d_data);
    return success ? 0 : 1;
}