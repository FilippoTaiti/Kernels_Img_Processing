#include "gpu.cuh"
__global__ void gpu_kernel(const uint8_t* input_data, int width, int height, int channels, int mask_width, const double* kernel, uint8_t* output_data, double normalizing_factor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < width && row < height && channel < channels) {
        int actual_column = col - mask_width/2;
        int actual_row = row - mask_width/2;


        double rgb = 0.0f;

        for (int k_col = 0; k_col < mask_width; k_col++) {
            for (int k_row = 0; k_row < mask_width; k_row++) {

                int real_col = actual_column + k_col;
                int real_row = actual_row + k_row;
                if (real_row >= 0 && real_row < width && real_col >= 0 && real_col < height) {
                    rgb += input_data[real_col*width*channels + real_row*channels + channel] * kernel
                       [k_col * mask_width + k_row];
                }

            }
        }
        if (rgb*normalizing_factor < 0.0) {
            output_data[col*width*channels + row*channels + channel] = 0.0;
        } else if (rgb*normalizing_factor > 255.0) {
            output_data[col*width*channels + row*channels + channel] = 255.0;
        } else {
            output_data[col*width*channels + row*channels + channel] = rgb*normalizing_factor;
        }

    }
}