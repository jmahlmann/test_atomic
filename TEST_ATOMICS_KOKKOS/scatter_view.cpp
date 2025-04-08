#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>

#define N 1000000

template <typename T>
double atomic_add_loop(const char* label) {
    Kokkos::Timer timer;
    Kokkos::View<T*> counter("Counter", 1);
    
    Kokkos::parallel_for("Atomic Add Loop", N, KOKKOS_LAMBDA(const int i) {
        const auto idx = Kokkos::atomic_fetch_add(&counter(0), 1);
    });
    
    Kokkos::fence();
    std::cout << "Time " << label << ": " << timer.seconds() << "s\n";
    return timer.seconds();
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Timing the atomic add for different data types
        atomic_add_loop<int>("Atomic Int");
        atomic_add_loop<int32_t>("Atomic Int32");
        atomic_add_loop<int64_t>("Atomic Int64");
        atomic_add_loop<long int>("Atomic LongInt");
        atomic_add_loop<unsigned long long int>("Atomic LongLongInt");
        atomic_add_loop<size_t>("Atomic SizeT");
        atomic_add_loop<double>("Atomic Double");
    }
    Kokkos::finalize();
    return 0;
}