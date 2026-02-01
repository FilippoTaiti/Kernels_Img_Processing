#include "gpu.cuh"

#include <algorithm>
#include <cstdio>

#include "utils.h"

__global__ void gpu_kernel(const uint8_t *input_data, const int width, const int height, const int channels,
                           const int mask_width, uint8_t *output_data, const double normalizing_factor) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    if (col < width && row < height && channel < channels) {
        const int actual_column = col - mask_width / 2;
        const int actual_row = row - mask_width / 2;

        double rgb = 0.0f;

        for (int k_row = 0; k_row < mask_width; k_row++) {
            for (int k_col = 0; k_col < mask_width; k_col++) {
                const int real_col = actual_column + k_col;
                const int real_row = actual_row + k_row;
                const int idx = (channel * width * height) + real_row * width + real_col;
                const int k_idx = k_row * mask_width + k_col;
                bool ipred = real_row >= 0 && real_row < height && real_col >= 0 && real_col < width;
                if (ipred) {
                    rgb += input_data[idx] * kernel[k_idx];
                }
            }
        }
        rgb *= normalizing_factor;
        const int out_idx = (channel * width * height) + row * width + col;
        output_data[out_idx] = static_cast<uint8_t>(fminf(fmaxf(rgb, 0.0f), 255.0f));
    }
}

__global__ void gpu_kernel_with_tiling(const uint8_t *input_data, const int width, const int height, const int channels,
                                       const int mask_width, uint8_t *output_data,
                                       const double normalizing_factor) {
    extern __shared__ uint8_t tile_p[];
    const int o_tile_width = TILE_WIDTH - (mask_width - 1);
    const int mask_radius = mask_width / 2;
    const int dim = TILE_WIDTH + mask_width - 1;

    const int output_row = blockIdx.y * o_tile_width + threadIdx.y;
    const int output_col = blockIdx.x * o_tile_width + threadIdx.x;

    const int input_row = output_row - mask_radius;
    const int input_col = output_col - mask_radius;

    for (int k = 0; k < channels; k++) {
        if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
            tile_p[k * dim * dim + threadIdx.y * dim + threadIdx.x] = input_data[
                k * width * height + input_row * width + input_col];
        } else {
            tile_p[k * dim * dim + threadIdx.y * dim + threadIdx.x] = 0.0;
        }
        __syncthreads();

        if (threadIdx.y < o_tile_width && threadIdx.x < o_tile_width) {
            double rgb = 0.0;
            for (int k_row = 0; k_row < mask_width; k_row++) {
                for (int k_col = 0; k_col < mask_width; k_col++) {
                    const int new_row = k_row + threadIdx.y;
                    const int new_col = k_col + threadIdx.x;
                    const int kernel_index = k_row * mask_width + k_col;
                    rgb += tile_p[k * dim * dim + new_row * dim + new_col] * kernel[kernel_index];
                }
            }
            if (output_row < height && output_col < width) {
                output_data[k * width * height + output_row * width + output_col] = static_cast<uint8_t>(fminf(fmaxf(rgb*normalizing_factor, 0.0f), 255.0f));
            }
        }
    }
}

void test_gpu(const Image &image, int mask_width, vector<double> &gpu_times, const bool tiling,
              const double normalizing_factor,
              uint8_t *output_data_gpu) {
    int total_bytes = (TILE_WIDTH + mask_width - 1) * (TILE_WIDTH + mask_width - 1) * image.channels * sizeof(uint8_t);
    const int o_tile_width = TILE_WIDTH - (mask_width - 1);
    const int dim = TILE_WIDTH + mask_width - 1;

    uint8_t *data_ptr;
    uint8_t *output_data;


    cudaError_t err = cudaMalloc(&data_ptr, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    err = cudaMalloc(&output_data, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));

    err = cudaMemcpy(data_ptr, image.data, image.size * sizeof(u_int8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));


    float ms = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (tiling) {
        dim3 dimBlock(dim, dim);
        dim3 dimGrid((image.width + o_tile_width - 1)/ o_tile_width,
                     (image.height + o_tile_width - 1) / o_tile_width, 1);
        for (int k = 0; k <= NUMBER_OF_ITERATIONS; k++) {
            if (k > 2) {
                cudaEventRecord(start);
                gpu_kernel_with_tiling<<<dimGrid, dimBlock, total_bytes>>>(
                    data_ptr, image.width, image.height, image.channels,
                    mask_width, output_data, normalizing_factor);


                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
                gpu_times[k - 3] = ms;
            } else {
                gpu_kernel_with_tiling<<<dimGrid, dimBlock, total_bytes>>>(
                    data_ptr, image.width, image.height, image.channels,
                    mask_width, output_data, normalizing_factor);

            }
        }
    } else {
        dim3 dimBlock(16, 16, 3);
        dim3 dimGrid(ceil((image.width + dimBlock.x - 1) / dimBlock.x),
                     ceil((image.height + dimBlock.y - 1) / dimBlock.y),
                     ceil((image.channels + dimBlock.z - 1) / dimBlock.z));
        for (int k = 0; k <= NUMBER_OF_ITERATIONS; k++) {
            if (k > 2) {
                cudaEventRecord(start);
                gpu_kernel<<<dimGrid, dimBlock>>>(data_ptr, image.width, image.height, image.channels,
                                                  mask_width, output_data, normalizing_factor);

                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
                gpu_times[k - 3] = ms;
            } else {
                gpu_kernel<<<dimGrid, dimBlock>>>(data_ptr, image.width, image.height, image.channels,
                                                  mask_width, output_data, normalizing_factor);

            }
        }
    }

    printf("Tempo di esecuzione GPU kernel %dx%d (ms): \n", mask_width, mask_width);
    printf("Min: %.4f\n", *min_element(gpu_times.begin(), gpu_times.end()));
    printf("Max: %.4f\n", *max_element(gpu_times.begin(), gpu_times.end()));
    const double avg = mean(gpu_times);
    printf("Avg: %.4f\n", avg);
    printf("Std: %.4f\n", standard_dev(gpu_times, avg));

    err = cudaMemcpy(output_data_gpu, output_data, image.size * sizeof(u_int8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));


    cudaFree(data_ptr);
    cudaFree(output_data);
}
