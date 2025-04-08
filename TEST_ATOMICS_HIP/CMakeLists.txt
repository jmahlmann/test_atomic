cmake_minimum_required(VERSION 3.16)

# Define the project name
project(HIPTutorialScatterView)

# Find the HIP package (ensure you have HIP installed and the environment variables are set)
find_package(HIP REQUIRED)

# Add the executable for the project
add_executable(scatterview scatter_view.cpp)

# Link the HIP device library for the executable
target_link_libraries(scatterview HIP::device)

# Set necessary compile options for HIP (optional, based on your needs)
# target_compile_options(scatterview PRIVATE -O3)  # Optimizations (optional)

# If needed, specify the target architecture for your GPU (adjust according to your specific GPU)
# if (CMAKE_CXX_COMPILER_ID STREQUAL "HIP")
target_compile_options(scatterview PRIVATE --amdgpu-target=gfx900)  # Adjust based on your GPU architecture
# endif()

# Make sure to use the HIP compiler (hipcc)
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})