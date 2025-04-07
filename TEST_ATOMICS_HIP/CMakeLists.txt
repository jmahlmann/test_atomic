cmake_minimum_required(VERSION 3.21)
project(HIPAtomicExample LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable HIP
find_package(HIP REQUIRED)

# Add executable
add_executable(scatter_view scatter_view.cpp)

# Link HIP to the executable
target_link_libraries(scatter_view PRIVATE HIP::device)