#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define identity3 {0, 0, 0, 0, 1, 0, 0, 0, 0}
#define identity5 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define edge0 {1, 0, -1, 0, 0, 0, 1, 0, 1}
#define edge1 {0, 1, 0, 1, -4, 1, 0, 1, 0}
#define edge2 {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}
#define sharpen {0, -1, 0, -1, 5, -1, 0, -1, 0}
#define box_blur {0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111}
#define gaussian_blur {1*0.0625, 2*0.0625, 1*0.0625, 2*0.0625, 4*0.0625, 2*0.0625, 1*0.0625, 2*0.0625, 1*0.0625}
#define emboss {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}

#define sobel3x3gx {-1, -2, -1, 0, 0, 0, 1, 2, 1}
#define sobel3x3gy {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}

#define sobel5x5gx {2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2}
#define sobel5x5gy {{2, 1, 0, -1, -2}, {2, 1, 0, -1, -2}, {4, 2, 0, -2, -4}, {2, 1, 0, -1, -2}, {2, 1, 0, -1, -2}}

#define NUMBER_OF_CHANNELS 3

using namespace std;

#include "cpu.h"
#include "utils.h"



int main() {
    Image test("dataset/n01693334_green_lizard.JPEG");
    float kernel[9] = edge1;



    auto* output_data = new uint8_t[test.size];
    applyKernel(test, 3, kernel, output_data);
    Image result3_image(test.width, test.height, test.channels, output_data);
    bool result = result3_image.writeImage("resultkernel33.png");

    float kernel25[25] = identity5;
/*
    applyKernel(test, 5, kernel25, output_data);
    Image result5_image(test.width, test.height, test.channels, output_data);
    result = result5_image.writeImage("resultkernel5.png");

*/






    return 0;
}
