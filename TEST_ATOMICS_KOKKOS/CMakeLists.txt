cmake_minimum_required(VERSION 3.16)
project(KokkosTutorialScatterView)
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
    add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
  else()
    include(FetchContent)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG        4.5.01
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

set(Kokkos_ENABLE_HIP
    ${default_KOKKOS_ENABLE_HIP}
    CACHE BOOL "Enable HIP")
set(Kokkos_ENABLE_CUDA
    ${default_KOKKOS_ENABLE_CUDA}
    CACHE BOOL "Enable CUDA")
set(Kokkos_ENABLE_OPENMP
    ${default_KOKKOS_ENABLE_OPENMP}
    CACHE BOOL "Enable OpenMP")

# set memory space
if(${Kokkos_ENABLE_CUDA})
  add_compile_definitions(CUDA_ENABLED)
  set(ACC_MEM_SPACE Kokkos::CudaSpace)
elseif(${Kokkos_ENABLE_HIP})
  add_compile_definitions(HIP_ENABLED)
  set(ACC_MEM_SPACE Kokkos::HIPSpace)
else()
  set(ACC_MEM_SPACE Kokkos::HostSpace)
endif()

add_executable(scatterview scatter_view.cpp)
target_link_libraries(scatterview Kokkos::kokkos)