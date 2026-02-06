cmake_minimum_required(VERSION 3.28)
project(untitled1 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.1/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -march=native -mtune=native -funroll-loops")




find_package(OpenCV REQUIRED)



add_executable(untitled1
        src/main.cu
        "src/cpu.cpp"
        "src/cpu.h"
        src/Utility/utils.cpp
        src/Utility/utils.h
        "src/gpu.cuh"
        "src/gpu.cu"

)

set_target_properties(untitled1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(untitled1 PRIVATE ${OpenCV_LIBS} cudart)

target_compile_options(untitled1 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
