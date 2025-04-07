#include<Kokkos_Core.hpp>
#include<Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <chrono>     

#include <iostream> // for std::cout

// Scatter Add algorithm using atomic add
double atomic_double_loop() {
Kokkos::Timer timer;
Kokkos::View<double*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_longint_loop() {
Kokkos::Timer timer;
Kokkos::View<long int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_longlongint_loop() {
Kokkos::Timer timer;
Kokkos::View<unsigned long long int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int_loop() {
Kokkos::Timer timer;
Kokkos::View<int*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int32_loop() {
Kokkos::Timer timer;
Kokkos::View<std::int32_t*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_int64_loop() {
Kokkos::Timer timer;
Kokkos::View<std::int64_t*> counter("Counter",1);
Kokkos::parallel_for("Atomic Loop", 1000000, 
 KOKKOS_LAMBDA(const int i) {
      const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
});
// Wait for Kernel to finish before timing
Kokkos::fence();
return timer.seconds();
}

// Scatter Add algorithm using atomic add
double atomic_sizet_loop() {
      Kokkos::Timer timer;
      Kokkos::View<std::int64_t*> counter("Counter",1);
      Kokkos::parallel_for("Atomic Loop", 1000000, 
       KOKKOS_LAMBDA(const int i) {
            const auto idx = Kokkos::atomic_fetch_add(&counter(0),1);
      });
      // Wait for Kernel to finish before timing
      Kokkos::fence();
      return timer.seconds();
      }

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {

      std::chrono::time_point<std::chrono::system_clock> start, end;
 
    start = std::chrono::system_clock::now();
    double time_loop = atomic_int_loop();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time Atomic Int: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_int32_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic Int32: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_int64_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic Int64: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_longint_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic LongInt: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_longlongint_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic LongLongInt: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_sizet_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic SizeT: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    time_loop = atomic_double_loop();
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Time Atomic Double: " << elapsed_seconds.count() << "s\n";


  }
  Kokkos::finalize();
}