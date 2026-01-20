//
// Created by filippo on 19/01/26.
//

#include "cpu.h"

void applyKernel(const Image &input_image, int mask_width, const float* kernel, uint8_t* output_data) {
    for (int k_roww = 0; k_roww < mask_width; k_roww++) {
        for (int k_coll = 0; k_coll < mask_width; k_coll++) {
            printf("K[%d][%d] = %f\n", k_roww, k_coll, kernel[k_roww * mask_width + k_coll]);
        }
    }
    for (int x = 0; x < input_image.width; x++) {
        for (int y = 0; y < input_image.height; y++) {
            for (int c = 0; c < input_image.channels; c++) {
                int actual_row = x - mask_width / 2;
                int actual_column = y - mask_width / 2;

                float rgb = 0;

                for (int k_row = 0; k_row < mask_width; k_row++) {
                    for (int k_col = 0; k_col < mask_width; k_col++) {

                        int cur_row = actual_row + k_row;
                        int cur_column = actual_column + k_col;
                        if (cur_row >= 0 && cur_row < input_image.width && cur_column >= 0 && cur_column < input_image.height) {
                            printf("%.2f + %d + %.2f\n", rgb, input_image.data[cur_row * input_image.width + cur_column], kernel
                            [k_row * mask_width + k_col]);
                            rgb += input_image.data[cur_column*input_image.width*input_image.channels + cur_row*input_image.channels + c] * kernel
                               [k_row * mask_width + k_col];
                        }


                    }
                    output_data[y*input_image.width*input_image.channels + x*input_image.channels + c] = clamp(static_cast<double>(fabs(rgb)), 0.0, 255.0);
                }

            }
        }
    }
}


/*void RGBtoGrayScale(const Image& input_image, uint8_t* output_data) {
    for (int x = 0; x < input_image.width; x++) {
        for (int y = 0; y < input_image.height; y++) {
            for (int c = 0; c < input_image.channels; c++) {
                output_data[x*input_image.width + y] = 0.2126* input_image.data[0*input_image.width*input_image.height + y*input_image.width + x]
                + 0.7152*input_image.data[1*input_image.width*input_image.height + y*input_image.width + x] + 0.0722 * input_image.data[2*input_image.width*input_image.height + y*input_image.width + x];
            }
        }
    }
}*/
