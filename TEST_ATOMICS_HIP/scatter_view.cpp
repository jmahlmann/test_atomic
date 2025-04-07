#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdint>

#define N 1000000

// Updated kernels with multiple threads per block
__global__ void atomicAdd_int_kernel(int* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1);
    }
}

__global__ void atomicAdd_int32_kernel(int32_t* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1);
    }
}

__global__ void atomicAdd_int64_kernel(int64_t* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(counter), 1ULL);
    }
}

__global__ void atomicAdd_long_kernel(long int* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(counter), 1ULL);
    }
}

__global__ void atomicAdd_ulonglong_kernel(unsigned long long int* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1ULL);
    }
}

__global__ void atomicAdd_double_kernel(double* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(counter, 1.0);
    }
}

__global__ void atomicAdd_size_t_kernel(size_t* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(reinterpret_cast<size_t*>(counter), 1ULL);
    }
}

template <typename T>
using KernelFunc = void (*)(T*);

template <typename T, KernelFunc<T> kernel>
double run_atomic(const char* label) {
    T* d_counter;
    hipMalloc(&d_counter, sizeof(T));
    hipMemset(d_counter, 0, sizeof(T));

    auto start = std::chrono::high_resolution_clock::now();

    // Launch a single kernel with enough threads to cover N operations
    int block_size = 256;  // Number of threads per block
    int num_blocks = (N + block_size - 1) / block_size;  // Total blocks required to cover N

    kernel<<<num_blocks, block_size>>>(d_counter);

    hipDeviceSynchronize();  // Wait for the kernel to finish

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time " << label << ": " << elapsed.count() << "s\n";

    hipFree(d_counter);
    return elapsed.count();
}

int main() {
    run_atomic<int, atomicAdd_int_kernel>("Atomic Int");
    run_atomic<int32_t, atomicAdd_int32_kernel>("Atomic Int32");
    run_atomic<int64_t, atomicAdd_int64_kernel>("Atomic Int64");
    run_atomic<long int, atomicAdd_long_kernel>("Atomic LongInt");
    run_atomic<unsigned long long int, atomicAdd_ulonglong_kernel>("Atomic LongLongInt");
    run_atomic<size_t, atomicAdd_size_t_kernel>("Atomic SizeT");
    run_atomic<double, atomicAdd_double_kernel>("Atomic Double");

    return 0;
}