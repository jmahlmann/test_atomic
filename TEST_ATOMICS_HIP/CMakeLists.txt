cmake_minimum_required(VERSION 3.21)
project(HIPAtomicBenchmark LANGUAGES CXX HIP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable HIP support
find_package(HIP REQUIRED)

# Optional: specify GPU target (e.g., gfx90a, gfx908, gfx1030)
# Uncomment and modify for your hardware
set(CMAKE_HIP_ARCHITECTURES "gfx90a")

# Add the executable
add_executable(scatter_view scatter_view.cpp)

# Link HIP device library
target_link_libraries(scatter_view PRIVATE HIP::device)