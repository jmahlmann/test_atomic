#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

#define N 1000000

// CUDA kernel for atomicAdd with int
__global__ void atomicAdd_int_kernel(int *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1);
    }
}

// CUDA kernel for atomicAdd with int32_t
__global__ void atomicAdd_int32_kernel(int32_t *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1);
    }
}

// CUDA kernel for atomicAdd with int64_t
__global__ void atomicAdd_int64_kernel(int64_t *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(counter), 1ULL);
    }
}

// CUDA kernel for atomicAdd with long int
__global__ void atomicAdd_long_kernel(long int *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(counter), 1ULL);
    }
}

// CUDA kernel for atomicAdd with unsigned long long int
__global__ void atomicAdd_ulonglong_kernel(unsigned long long int *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1ULL);
    }
}

// CUDA kernel for atomicAdd with double
__global__ void atomicAdd_double_kernel(double *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1.0);
    }
}

// Template function to run atomic test and check correctness
template <typename T, void (*kernel)(T *)>
double run_atomic(const char *label) {
    T *d_counter;
    T h_counter = 0;

    cudaMalloc(&d_counter, sizeof(T));
    cudaMemset(d_counter, 0, sizeof(T));

    auto start = std::chrono::high_resolution_clock::now();

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    kernel<<<num_blocks, block_size>>>(d_counter);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(&h_counter, d_counter, sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "Time " << label << ": " << elapsed.count() << "s\n";
    std::cout << "Result for " << label << ": " << h_counter;

    bool correct = (static_cast<uint64_t>(h_counter) == static_cast<uint64_t>(N));
    std::cout << (correct ? " [PASS]" : " [FAIL] Expected " + std::to_string(N)) << "\n\n";

    cudaFree(d_counter);
    return elapsed.count();
}

int main() {
    run_atomic<int, atomicAdd_int_kernel>("Atomic Int");
    run_atomic<int32_t, atomicAdd_int32_kernel>("Atomic Int32");
    run_atomic<int64_t, atomicAdd_int64_kernel>("Atomic Int64");
    run_atomic<long int, atomicAdd_long_kernel>("Atomic LongInt");
    run_atomic<unsigned long long int, atomicAdd_ulonglong_kernel>("Atomic LongLongInt");
    run_atomic<double, atomicAdd_double_kernel>("Atomic Double");

    return 0;
}