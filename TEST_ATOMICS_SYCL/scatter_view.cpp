#include <CL/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>

#define N 1000000

// Templated struct to generate unique kernel names
template <typename T>
struct AtomicAddKernel;

template <typename T>
double run_atomic(cl::sycl::queue &q, const char *label) {
    T *counter = cl::sycl::malloc_device<T>(1, q);
    q.memset(counter, 0, sizeof(T)).wait();

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](cl::sycl::handler &h) {
        h.parallel_for<AtomicAddKernel<T>>(cl::sycl::range<1>(N), [=](cl::sycl::id<1> i) {
            cl::sycl::atomic_ref<T,
                                 cl::sycl::memory_order::relaxed,
                                 cl::sycl::memory_scope::device,
                                 cl::sycl::access::address_space::global_space>
                atomic_counter(*counter);
            atomic_counter.fetch_add(1);
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time " << label << ": " << elapsed.count() << "s\n";

    cl::sycl::free(counter, q);
    return elapsed.count();
}

int main() {
    cl::sycl::queue q(cl::sycl::default_selector{});
    std::cout << "Running on device: "
              << q.get_device().get_info<cl::sycl::info::device::name>()
              << "\n";

    run_atomic<int>(q, "Atomic Int");
    run_atomic<int32_t>(q, "Atomic Int32");
    run_atomic<int64_t>(q, "Atomic Int64");
    run_atomic<long int>(q, "Atomic LongInt");
    run_atomic<unsigned long int>(q, "Atomic ULongInt");
    run_atomic<long long int>(q, "Atomic LongLong");
    run_atomic<unsigned long long int>(q, "Atomic ULongLong");
    run_atomic<size_t>(q, "Atomic SizeT");
    run_atomic<double>(q, "Atomic Double");
    run_atomic<float>(q, "Atomic Double");

    return 0;
}