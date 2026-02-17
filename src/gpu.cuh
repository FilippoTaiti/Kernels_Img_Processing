#include <cstdint>
#include <vector>
#define MAX_MASK_WIDTH 11
__constant__ inline float kernel[MAX_MASK_WIDTH*MAX_MASK_WIDTH];

#include "Utility/utils.h"
using namespace std;
#include <string>
#define TILE_WIDTH 16

__global__ void gpuKernelPlanarStream(const uint8_t* input_data, int width, int height, int mask_width, uint8_t* output_data);
__global__ void gpuKernelPlanarTilingStream(const uint8_t* input_data, int width, int height, int mask_width, uint8_t* output_data);
__global__ void gpuKernelInterleaved(const uint8_t* input_data, int width, int height, int channels, int mask_width, uint8_t* output_data);
__global__ void gpuKernelInterleavedTiling(const uint8_t* input_data, int width, int height, int mask_width, uint8_t* output_data);
__global__ void gpuKernelPlanar(const uint8_t* input_data, int width, int height, int channels, int mask_width, uint8_t* output_data);
__global__ void gpuKernelPlanarTiling(const uint8_t* input_data, int width, int height, int mask_width, uint8_t* output_data);


#ifdef __cplusplus
extern "C" {
    #endif
    void testGPUStreamTilingVSNoTiling(const Image& image, int mask_width, vector<float>& gpu_times, bool tiling, uint8_t* output_data_gpu);
    #ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
    void testGPUInterleavedVSPlanar(const Image &image, const int mask_width, vector<float> &gpu_times, const bool interleaved, const bool tiling,
              uint8_t *output_data_gpu);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    void testWrapperStreamTilingVSNoTiling(const Image& image, vector<float>& cpu_times, vector<float>& gpu_times, string name, float* kernel3x3, float* kernel7x7, float* kernel11x11);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    void testWrapperInterleavedVSPlanar(const Image &image, vector<float> &cpu_times, vector<float> &gpu_times, string name, float *kernel3x3,
                  float *kernel7x7, float *kernel11x11);
#ifdef __cplusplus
}
#endif
