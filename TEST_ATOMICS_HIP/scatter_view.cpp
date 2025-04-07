#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>
#include <cstdint>

#define N 1000000

__global__ void atomicAdd_int(int* counter) {
    atomicAdd(counter, 1);
}

__global__ void atomicAdd_int32(std::int32_t* counter) {
    atomicAdd(counter, 1);
}

__global__ void atomicAdd_int64(std::int64_t* counter) {
    atomicAdd((unsigned long long int*)counter, 1ULL); // HIP workaround
}

__global__ void atomicAdd_long(long int* counter) {
    atomicAdd((unsigned long long int*)counter, 1ULL); // HIP workaround
}

__global__ void atomicAdd_ulonglong(unsigned long long int* counter) {
    atomicAdd(counter, 1ULL);
}

__global__ void atomicAdd_double(double* counter) {
    atomicAdd(counter, 1.0);
}

template <typename T, void (*Kernel)(T*)>
double run_atomic_test(const char* label) {
    T* d_counter;
    hipMalloc(&d_counter, sizeof(T));
    hipMemset(d_counter, 0, sizeof(T));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        Kernel<<<1,1>>>(d_counter);
    }
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time " << label << ": " << elapsed.count() << "s\n";

    hipFree(d_counter);
    return elapsed.count();
}

int main() {
    run_atomic_test<int, atomicAdd_int>("Atomic Int");
    run_atomic_test<std::int32_t, atomicAdd_int32>("Atomic Int32");
    run_atomic_test<std::int64_t, atomicAdd_int64>("Atomic Int64");
    run_atomic_test<long int, atomicAdd_long>("Atomic LongInt");
    run_atomic_test<unsigned long long int, atomicAdd_ulonglong>("Atomic LongLongInt");
    run_atomic_test<double, atomicAdd_double>("Atomic Double");

    return 0;
}
