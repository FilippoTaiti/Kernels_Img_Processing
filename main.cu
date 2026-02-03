#include <iostream>
#include "cpu.h"
#include "utils.h"
#include "gpu.cuh"

using namespace std;

int main() {
    vector<float> cpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);
    vector<float> gpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);

    float kernel7x7[49];
    generateGaussianKernel(7, 1.4, kernel7x7);
    float kernel11x11[121];
    generateGaussianKernel(11, 2, kernel11x11);

    printf("Small image : \n");
    Image small_image("dataset/n01693334_green_lizard.JPEG");
    small_image.data = toPlanar(small_image);

    auto* output_data_cpu_small = new uint8_t[small_image.size];
    auto* output_data_gpu_small = new uint8_t[small_image.size];

    test_wrapper(small_image, cpu_times, gpu_times, output_data_cpu_small, output_data_gpu_small, "small");

    printf("\nImmagine 2K : \n");
    Image image2K("dataset/pexels-covandenham-1108753.jpg");
    image2K.data = toPlanar(image2K);

    auto* output_data_cpu_2K = new uint8_t[image2K.size];
    auto* output_data_gpu_2K = new uint8_t[image2K.size];

    test_wrapper(image2K, cpu_times, gpu_times, output_data_cpu_2K, output_data_gpu_2K, "2K");


    printf("\nImmagine 4K : \n");
    Image image4K("dataset/pexels-zelch-12498925.jpg");
    image4K.data = toPlanar(image4K);
    auto* output_data_cpu_4K= new uint8_t[image4K.size];
    auto* output_data_gpu_4K = new uint8_t[image4K.size];

    test_wrapper(image4K, cpu_times, gpu_times, output_data_cpu_4K, output_data_gpu_4K, "4K");

    return 0;
}
