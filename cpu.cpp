//
// Created by filippo on 19/01/26.
//

#include "cpu.h"

void applyKernelPlanar(const Image &input_image, int mask_width, const float *kernel, uint8_t *output_data) {
    for (int x = 0; x < input_image.width; x++) {
        for (int y = 0; y < input_image.height; y++) {
            for (int c = 0; c < input_image.channels; c++) {
                const int actual_row = y - mask_width / 2;
                const int actual_column = x - mask_width / 2;

                float rgb = 0.0f;

                for (int k_col = 0; k_col < mask_width; k_col++) {
                    for (int k_row = 0; k_row < mask_width; k_row++) {
                        int cur_row = actual_row + k_row;
                        int cur_column = actual_column + k_col;
                        if (cur_row >= 0 && cur_row < input_image.height && cur_column >= 0 && cur_column <
                            input_image.width) {
                            rgb += input_image.data[
                                        (c * input_image.width * input_image.height) + cur_row * input_image.
                                        width + cur_column] * kernel
                                    [k_col * mask_width + k_row];
                        }
                    }
                    output_data[(c * input_image.width * input_image.height) + y * input_image.width + x] = static_cast<uint8_t>(clamp(rgb, 0.0f, 255.0f));
                }
            }
        }
    }
}

void test_cpu(const Image &input_image, const int mask_width, const float* kernel, uint8_t* output_data, vector<float>& cpu_times) {
    for (int k = 0; k < NUMBER_OF_ITERATIONS; k++) {
        if (k > 1) {
            clock_t start1 = clock();
            auto start = chrono::high_resolution_clock::now();
            applyKernelPlanar(input_image, mask_width, kernel, output_data);
            auto end = chrono::high_resolution_clock::now();
            clock_t end1 = clock();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            double duration_cpu_time = ((double) (end1-start1)) / CLOCKS_PER_SEC;

            cpu_times[k - 2] = duration.count();

        } else {
            applyKernelPlanar(input_image, mask_width, kernel, output_data);
        }
    }

    printf("Tempi di esecuzione CPU kernel %dx%d (ms):\n", mask_width, mask_width);
    printf("Min: %.4f\n", *min_element(cpu_times.begin(), cpu_times.end()));
    printf("Max: %.4f\n", *max_element(cpu_times.begin(), cpu_times.end()));
    const float avg = mean(cpu_times);
    printf("Avg: %.4f\n", avg);
    printf("Std: %.4f\n", standard_dev(cpu_times, avg));
}
