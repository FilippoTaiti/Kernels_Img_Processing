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






int main() {
    Image image("dataset/n01693334_green_lizard.JPEG");
    auto* output_data = new uint8_t[image.size];

    double kernel3x3[9] = gaussian3x3;
    double sum = 0.0f;
    for (int k = 0; k < 15; k++) {
        if (k > 4) {
            auto start = chrono::high_resolution_clock::now();
            applyKernel(image, 3, kernel3x3, output_data, 0.0625);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            sum += duration.count();


        }

    }
    printf("Tempo di esecuzione kernel 3x3 (ms): %.2f\n", sum);
    Image result3_image(image.width, image.height, image.channels, output_data);
    bool result = result3_image.writeImage("result_gaussian_kernel_3X3.png");


    double kernel5x5[25] = gaussian5x5;
    sum = 0.0f;
    for (int k = 0; k < 15; k++) {
        if (k > 4) {
            auto start = chrono::high_resolution_clock::now();
            applyKernel(image, 5, kernel5x5, output_data, 0.0036630037);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            sum += duration.count();
        }

    }
    printf("Tempo di esecuzione kernel 5x5 (ms): %.2f\n", sum);
    Image result5_image(image.width, image.height, image.channels, output_data);
    result = result5_image.writeImage("result_gaussian_kernel_5X5.png");


    double kernel7x7[49] = gaussian7x7;
    sum = 0.0f;
    for (int k = 0; k < 15; k++) {
        if (k > 4) {
            auto start = chrono::high_resolution_clock::now();
            applyKernel(image, 7, kernel7x7, output_data, 0.000997009);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            sum += duration.count();
        }

    }
    printf("Tempo di esecuzione kernel 7x7 (ms): %.2f", sum);
    Image result7_image(image.width, image.height, image.channels, output_data);
    result = result7_image.writeImage("result_gaussian_kernel_7X7.png");



    return 0;
}
