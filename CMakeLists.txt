cmake_minimum_required(VERSION 3.28)
project(untitled1 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.1/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120)





find_package(OpenCV REQUIRED)



add_executable(untitled1
        main.cu
        cpu.cpp
        cpu.h
        utils.cpp
        utils.h

)

set_target_properties(untitled1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

