//
// Created by filippo on 19/01/26.
//

#include "cpu.h"

void applyKernel(const Image &input_image, int mask_width, const double* kernel, uint8_t* output_data, double normalizing_factor) {
    for (int y = 0; y < input_image.height; y++) {
        for (int x = 0; x < input_image.width; x++) {
            for (int c = 0; c < input_image.channels; c++) {
                int actual_row = x - mask_width / 2;
                int actual_column = y - mask_width / 2;

                double rgb = 0.0f;

                for (int k_col = 0; k_col < mask_width; k_col++) {
                    for (int k_row = 0; k_row < mask_width; k_row++) {

                        int cur_row = actual_row + k_row;
                        int cur_column = actual_column + k_col;
                        if (cur_row >= 0 && cur_row < input_image.width && cur_column >= 0 && cur_column < input_image.height) {
                            rgb += input_image.data[cur_column*input_image.width*input_image.channels + cur_row*input_image.channels + c] * kernel
                               [k_col * mask_width + k_row];
                        }


                    }
                    output_data[y*input_image.width*input_image.channels + x*input_image.channels + c] = clamp(rgb*normalizing_factor, 0.0, 255.0);
                }

            }
        }
    }
}
