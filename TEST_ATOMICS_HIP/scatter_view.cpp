#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>

#define N 1000000

__global__ void atomicAdd_int_kernel(int* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1);
    }
} 

__global__ void atomicAdd_int32_kernel(int32_t* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1);
    }
}

__global__ void atomicAdd_int64_kernel(int64_t* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1LL);
    }
}

__global__ void atomicAdd_longint_kernel(long int* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd((unsigned long long int*)counter, 1ULL); // HIP workaround
    }
}

__global__ void atomicAdd_ulonglong_kernel(unsigned long long int* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1ULL);
    }
}

__global__ void atomicAdd_double_kernel(double* counter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1.0);
    }
}

void run_and_time(void (*kernel)(void*), void* counter, size_t size, const char* label, hipStream_t stream) {
    hipMemset(counter, 0, size);
    auto start = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads, 0, stream>>>(counter);
    hipStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time " << label << ": " << elapsed.count() << "s\n";
}

int main() {
    hipStream_t stream;
    hipStreamCreate(&stream);

    int* d_int; hipMalloc(&d_int, sizeof(int));
    int32_t* d_int32; hipMalloc(&d_int32, sizeof(int32_t));
    int64_t* d_int64; hipMalloc(&d_int64, sizeof(int64_t));
    long int* d_long; hipMalloc(&d_long, sizeof(long int));
    unsigned long long int* d_ull; hipMalloc(&d_ull, sizeof(unsigned long long int));
    double* d_double; hipMalloc(&d_double, sizeof(double));

    run_and_time((void (*)(void*))atomicAdd_int_kernel, d_int, sizeof(int), "Atomic Int", stream);
    run_and_time((void (*)(void*))atomicAdd_int32_kernel, d_int32, sizeof(int32_t), "Atomic Int32", stream);
    run_and_time((void (*)(void*))atomicAdd_int64_kernel, d_int64, sizeof(int64_t), "Atomic Int64", stream);
    run_and_time((void (*)(void*))atomicAdd_longint_kernel, d_long, sizeof(long int), "Atomic LongInt", stream);
    run_and_time((void (*)(void*))atomicAdd_ulonglong_kernel, d_ull, sizeof(unsigned long long int), "Atomic LongLongInt", stream);
    run_and_time((void (*)(void*))atomicAdd_double_kernel, d_double, sizeof(double), "Atomic Double", stream);

    hipFree(d_int);
    hipFree(d_int32);
    hipFree(d_int64);
    hipFree(d_long);
    hipFree(d_ull);
    hipFree(d_double);

    hipStreamDestroy(stream);
    return 0;
}