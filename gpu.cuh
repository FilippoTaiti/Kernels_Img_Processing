#include <cstdint>
#include <vector>
#define MAXMASKWIDTH 7
__constant__ inline double kernel[MAXMASKWIDTH*MAXMASKWIDTH];

#include "utils.h"
using namespace std;
#define TILE_WIDTH 16

__global__ void gpu_kernel(const uint8_t* input_data, int width, int height, int channels, int mask_width, uint8_t* output_data, double normalizing_factor);
__global__ void gpu_kernel_with_tiling(const uint8_t* input_data, int width, int height, int channels, int mask_width, uint8_t* output_data, double normalizing_factor);



#ifdef __cplusplus
extern "C" {
    #endif
    void test_gpu(const Image& image, int mask_width, vector<double>& gpu_times, bool tiling, double normalizing_factor, uint8_t* output_data_gpu);
    #ifdef __cplusplus
}
#endif
