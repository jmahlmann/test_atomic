# Define the minimum required CMake version
cmake_minimum_required(VERSION 3.16)

# Define the project name and version
project(HIPTutorialScatterView VERSION 1.0 LANGUAGES CXX HIP)

# Get ROCm CMake Helpers onto your CMake Module Path
enable_language(HIP)

# Check if ROCM_PATH is set, either from the environment or from a cached path
if (NOT DEFINED ROCM_PATH)
  if (NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm path")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm path")
  endif()
endif()

# Add ROCm CMake module path
set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake" ${CMAKE_MODULE_PATH})

# Find the required HIP components
find_package(HIP REQUIRED)

# If necessary, you can also link other libraries like hipfft, hiprand, rocrand, etc.
# Here we're just linking with HIP
find_package(rocprim REQUIRED)

# Define your source files here (replace with actual source files for your program)
set(MY_HIP_SRCS scatter_view.cpp)  # Include your actual source file(s)

# Mark source files as HIP code
set_source_files_properties(${MY_HIP_SRCS} PROPERTIES LANGUAGE HIP)

# Add the executable that will use HIP
add_executable(scatterview ${MY_HIP_SRCS})

# Link necessary libraries
target_link_libraries(scatterview PRIVATE
  roc::rocprim_hip
)

# If you want to specify the target architecture for GPU, you can set flags
# Adjust the `--amdgpu-target=gfx900` flag as needed for your GPU architecture
target_compile_options(scatterview PRIVATE --amdgpu-target=gfx900)  # Example for gfx900, change to your GPU architecture

# Ensure to use the HIP compiler (hipcc)
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

# Set other compilation options if necessary
# target_compile_options(scatterview PRIVATE -O3)  # For optimizations (optional)