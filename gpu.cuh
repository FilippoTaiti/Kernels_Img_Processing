#include <cstdint>
__global__ void gpu_kernel(const uint8_t* input_data, int width, int height, int channels, int mask_width, const double* kernel, uint8_t* output_data, double normalizing_factor);