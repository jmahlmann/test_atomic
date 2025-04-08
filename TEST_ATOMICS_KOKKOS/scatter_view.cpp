#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>

#define N 1000000

template <typename T>
double atomic_add_loop(const char* label) {
    Kokkos::View<T*> counter("Counter", 1);
    Kokkos::deep_copy(counter, T(0));  // Ensure initialized to zero

    Kokkos::Timer timer;

    Kokkos::parallel_for("Atomic Add Loop", N, KOKKOS_LAMBDA(const int i) {
        Kokkos::atomic_fetch_add(&counter(0), T(1));
    });

    Kokkos::fence();
    double elapsed = timer.seconds();

    T result;
    Kokkos::deep_copy(result, counter);

    std::cout << "Time " << label << ": " << elapsed << "s\n";
    std::cout << "Result for " << label << ": " << result;

    bool correct = (static_cast<uint64_t>(result) == static_cast<uint64_t>(N));
    std::cout << (correct ? " [PASS]" : " [FAIL] Expected " + std::to_string(N)) << "\n\n";

    return elapsed;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
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