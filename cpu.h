//
// Created by filippo on 19/01/26.
//

#ifndef UNTITLED1_CPU_H
#define UNTITLED1_CPU_H

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include "utils.h"
#include <time.h>


void applyKernelPlanar(const Image& input_image, int mask_width, const float* kernel, uint8_t* output_data);


void test_cpu(const Image& input_image, int mask_width, const float* kernel, uint8_t *output_data, vector<float>& cpu_times);


#endif //UNTITLED1_CPU_H
