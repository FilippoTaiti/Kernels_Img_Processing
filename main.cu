#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// https://www.researchgate.net/figure/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7_fig2_325768087


#define identity3 {0, 0, 0, 0, 1, 0, 0, 0, 0}
#define identity5 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define gaussian_blur3x3 {1, 2, 1, 2, 4, 2, 1, 2, 1}
#define gaussian_blur5x5 {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1}
#define ridge {0, -1, 0, -1, 4, -1, 0, -1, 0}
#define edge {-1, -1, -1, -1, 8, -1, -1, -1, -1}
#define sharpen {0, -1, 0, -1, 5, -1, 0, -1, 0}
#define box_blur {0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111}
#define gaussian3x3 {1, 2, 1, 2, 4, 2, 1, 2, 1}
#define gaussian5x5 {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1}
#define gaussian7x7 {0, 0, 1, 2, 1, 0, 0, 0, 3, 13, 22, 13, 3, 0, 1, 13, 59, 97, 59, 13, 1, 2, 22, 97, 159, 97, 22, 2, 1, 13, 59, 97, 59, 13, 1, 0, 3, 13, 22, 13, 3, 0, 0, 0, 1, 2, 1, 0, 0}

#define NUMBER_OF_CHANNELS 3

using namespace std;

#include "cpu.h"
#include "utils.h"
#include "gpu.cuh"


int main() {


    Image image("dataset/n01693334_green_lizard.JPEG");
    image.data = toPlanar(image);

    auto *output_data = new uint8_t[image.size];


    vector<double> cpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);
    vector<double> gpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);

    double kernel3x3[9] = gaussian3x3;
    double kernel5x5[25] = gaussian5x5;
    double kernel7x7[49] = gaussian7x7;

    printf("CPU...\n");


    int mask_width = 3;

    auto* output_data_planar = new uint8_t[image.size];
    test_cpu(image, mask_width, kernel3x3, output_data_planar, 0.0625, cpu_times);
    Image result3_image_cpu(image.width, image.height, image.channels, output_data_planar);
    result3_image_cpu.data = toInterleaved(result3_image_cpu);
    bool result = result3_image_cpu.writeImage("result_gaussian_kernel_3X3_cpu.png");

    mask_width = 5;

    test_cpu(image, mask_width, kernel5x5, output_data_planar, 0.0036630037, cpu_times);
    Image result5_image_cpu(image.width, image.height, image.channels, output_data_planar);
    result5_image_cpu.data = toInterleaved(result5_image_cpu);
    result = result5_image_cpu.writeImage("result_gaussian_kernel_5X5_cpu.png");

    mask_width = 7;

    test_cpu(image, mask_width, kernel7x7, output_data_planar, 0.000997009, cpu_times);
    Image result7_image_cpu(image.width, image.height, image.channels, output_data_planar);
    result7_image_cpu.data = toInterleaved(result7_image_cpu);
    result = result7_image_cpu.writeImage("result_gaussian_kernel_7X7_cpu.png");


    printf("\nGPU with tiling...\n");

    mask_width = 3;

    cudaError_t err = cudaMemcpyToSymbol(kernel, kernel3x3, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, 0.0625, output_data);
    Image result3_gpu_image(image.width, image.height, image.channels, output_data);
    result3_gpu_image.data = toInterleaved(result3_gpu_image);
    result = result3_gpu_image.writeImage("result_gaussian_kernel_3X3_gpu_tiling.png");

    mask_width = 5;

    err = cudaMemcpyToSymbol(kernel, kernel5x5, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, 0.0036630037, output_data);
    Image result5_gpu_image(image.width, image.height, image.channels, output_data);
    result5_gpu_image.data = toInterleaved(result5_gpu_image);
    result = result5_gpu_image.writeImage("result_gaussian_kernel_5X5_gpu_tiling.png");

    mask_width = 7;

    err = cudaMemcpyToSymbol(kernel, kernel7x7, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, 0.000997009, output_data);
    Image result7_gpu_image(image.width, image.height, image.channels, output_data);
    result7_gpu_image.data = toInterleaved(result7_gpu_image);
    result = result7_gpu_image.writeImage("result_gaussian_kernel_7X7_gpu_tiling.png");


    printf("\nGPU without tiling...\n");

    mask_width = 3;

    err = cudaMemcpyToSymbol(kernel, kernel3x3, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, false, 0.0625, output_data);
    Image result3_gpu_image_tiling(image.width, image.height, image.channels, output_data);
    /*for (int row = 0; row < result3_gpu_image_tiling.height; row++) {
        for (int col = 0; col < result3_gpu_image_tiling.width; col++) {
            printf("%d\n", result3_gpu_image_tiling.data[row*result3_gpu_image_tiling.width+col]);
        }
    }*/
    result3_gpu_image_tiling.data = toInterleaved(result3_gpu_image_tiling);
    result = result3_gpu_image_tiling.writeImage("result_gaussian_kernel_3X3_gpu.png");

    mask_width = 5;

    err = cudaMemcpyToSymbol(kernel, kernel5x5, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, false, 0.0036630037, output_data);
    Image result5_gpu_image_tiling(image.width, image.height, image.channels, output_data);
    result5_gpu_image_tiling.data = toInterleaved(result5_gpu_image_tiling);
    result = result5_gpu_image_tiling.writeImage("result_gaussian_kernel_5X5_gpu.png");

    mask_width = 7;

    err = cudaMemcpyToSymbol(kernel, kernel7x7, mask_width * mask_width * sizeof(double));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, false, 0.000997009, output_data);
    Image result7_gpu_image_tiling(image.width, image.height, image.channels, output_data);
    result7_gpu_image_tiling.data = toInterleaved(result7_gpu_image_tiling);
    result = result7_gpu_image_tiling.writeImage("result_gaussian_kernel_7X7_gpu.png");


    return 0;
}
