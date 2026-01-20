//
// Created by filippo on 19/01/26.
//

#ifndef UNTITLED1_CPU_H
#define UNTITLED1_CPU_H

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include "utils.h"

void applyKernel(const Image& input_image, int mask_width, const double* kernel, uint8_t* output_data, double normalizing_factor);

#endif //UNTITLED1_CPU_H