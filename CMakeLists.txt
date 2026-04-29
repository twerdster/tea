cmake_minimum_required(VERSION 3.24)

project(Tea LANGUAGES CXX CUDA)

option(TEA_ENABLE_OPENMP "Enable OpenMP for host-side parallel loops" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile for")
endif()

add_executable(
    Tea
    histograms.cu
    entropies.cu
    utilities.cu
    TreeBuilder.cu
)

target_include_directories(Tea PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    Tea
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LINKER_LANGUAGE CUDA
)

if(TEA_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_compile_options(Tea PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-fopenmp>")
        target_link_options(Tea PRIVATE "$<$<LINK_LANGUAGE:CUDA>:-fopenmp>")
    else()
        message(STATUS "OpenMP was requested but not found; building without OpenMP")
    endif()
endif()
