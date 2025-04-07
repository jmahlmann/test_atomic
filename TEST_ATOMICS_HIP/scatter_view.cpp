#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdint>

#define N 1000000

__global__ void atomicAdd_int_kernel(int* counter) {
    atomicAdd(counter, 1);
}

__global__ void atomicAdd_int32_kernel(int32_t* counter) {
    atomicAdd(counter, 1);
}

__global__ void atomicAdd_int64_kernel(int64_t* counter) {
    atomicAdd(reinterpret_cast<unsigned long long int*>(counter), 1ULL);  
}

__global__ void atomicAdd_long_kernel(long int* counter) {
    atomicAdd(reinterpret_cast<unsigned long long int*>(counter), 1ULL);
}

__global__ void atomicAdd_ulonglong_kernel(unsigned long long int* counter) {
    atomicAdd(counter, 1ULL);
}

__global__ void atomicAdd_double_kernel(double* counter) {
    atomicAdd(counter, 1.0);
}

__global__ void atomicAdd_size_t_kernel(size_t* counter) {
    atomicAdd(reinterpret_cast<size_t*>(counter), 1ULL);
}

template <typename T>
using KernelFunc = void (*)(T*);

template <typename T, KernelFunc<T> kernel>
double run_atomic(const char* label) {
    T* d_counter;
    hipMalloc(&d_counter, sizeof(T));
    hipMemset(d_counter, 0, sizeof(T));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        kernel<<<1, 1>>>(d_counter);
    }

    hipDeviceSynchronize();

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
