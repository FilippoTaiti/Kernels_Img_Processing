#include <iostream>
#include "cpu.h"
#include "utils.h"
#include "gpu.cuh"

using namespace std;

int main() {
    vector<float> cpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);
    vector<float> gpu_times((NUMBER_OF_ITERATIONS - 2), 0.0f);

    float kernel3x3[9];
    generateGaussianKernel(3, 0.8, kernel3x3);
    float kernel7x7[49];
    generateGaussianKernel(7, 1.4, kernel7x7);
    float kernel11x11[121];
    generateGaussianKernel(11, 2, kernel11x11);

    printf("Immagine piccola : \n");
    Image image1K("dataset/pexels-snapwire-186566.jpg");
    image1K.data = toPlanar(image1K);


    test_wrapper(image1K, cpu_times, gpu_times,  "small", kernel3x3, kernel7x7, kernel11x11);

    printf("\nImmagine 2K : \n");
    Image image2K("dataset/pexels-covandenham-1108753.jpg");
    image2K.data = toPlanar(image2K);

    test_wrapper(image2K, cpu_times, gpu_times, "2K", kernel3x3, kernel7x7, kernel11x11);


    printf("\nImmagine 4K : \n");
    Image image4K("dataset/pexels-zelch-12498925.jpg");
    image4K.data = toPlanar(image4K);

    test_wrapper(image4K, cpu_times, gpu_times, "4K", kernel3x3, kernel7x7, kernel11x11);

    return 0;
}
