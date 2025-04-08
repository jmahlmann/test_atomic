# Minimum required CMake version
cmake_minimum_required(VERSION 3.18)  # 3.18+ has better CUDA support

# Define the project
project(CUDATutorialScatterView VERSION 1.0 LANGUAGES CXX CUDA)

# Enable CUDA language
enable_language(CUDA)

# Define your CUDA source files
set(MY_CUDA_SRCS scatter_view.cu)  # Replace with your actual source files

# Add the executable
add_executable(scatterview ${MY_CUDA_SRCS})

# Set the target architecture for GH200 (Hopper architecture, compute capability 9.0)
# Adjust if you want PTX fallback for future compatibility
set_target_properties(scatterview PROPERTIES
  CUDA_ARCHITECTURES "90"
)

# Optional: Enable fast math and optimizations
target_compile_options(scatterview PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Optional: If you're using C++17 or newer
target_compile_features(scatterview PRIVATE cxx_std_17)

# If you use any additional CUDA libraries like cuRAND, cuFFT, etc., you can link them here
# For example: target_link_libraries(scatterview PRIVATE cudart)