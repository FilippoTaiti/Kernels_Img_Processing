#include "gpu.cuh"

#include <algorithm>
#include <cstdio>

#include "cpu.h"
#include "utils.h"

__global__ void gpu_kernel(const uint8_t *input_data, const int width, const int height, const int channels,
                           const int mask_width, uint8_t *output_data) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        const int actual_column = col - mask_width / 2;
        const int actual_row = row - mask_width / 2;


        float rgb = 0.0f;

        for (int k_row = 0; k_row < mask_width; k_row++) {
            for (int k_col = 0; k_col < mask_width; k_col++) {
                const int real_col = actual_column + k_col;
                const int real_row = actual_row + k_row;
                const int k_idx = k_row * mask_width + k_col;
                bool ipred = real_row >= 0 && real_row < height && real_col >= 0 && real_col < width;
                if (ipred) {
                    rgb += input_data[real_row * width + real_col] * kernel[k_idx];
                }
            }
        }
        const int out_idx = row * width + col;
        output_data[out_idx] = static_cast<uint8_t>(min(max(rgb, 0.0f), 255.0f));
    }
}

__global__ void gpu_kernel_with_tiling(const uint8_t *input_data, const int width, const int height, const int channels,
                                       const int mask_width, uint8_t *output_data) {
    extern __shared__ uint8_t tile_p[];
    const int mask_radius = mask_width / 2;
    const int dim = TILE_WIDTH + mask_width - 1;

    const int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    const int all_pixels = dim * dim;

    for (int i = dest; i < all_pixels; i += TILE_WIDTH * TILE_WIDTH) {
        const int destY = i / dim;
        const int destX = i % dim;

        const int srcY = blockIdx.y * TILE_WIDTH + destY - mask_radius;
        const int srcX = blockIdx.x * TILE_WIDTH + destX - mask_radius;

        const int src = srcY * width + srcX;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            tile_p[destY * dim + destX] = input_data[src];
        } else {
            tile_p[destY * dim + destX] = 0;
        }
    }

    __syncthreads();

    const int output_row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int output_col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    if (output_row < height && output_col < width) {
        float rgb = 0.0f;
        for (int k_row = 0; k_row < mask_width; k_row++) {
            for (int k_col = 0; k_col < mask_width; k_col++) {
                int new_row = threadIdx.y + k_row;
                int new_col = threadIdx.x + k_col;
                rgb += tile_p[new_row * dim + new_col] * kernel[k_row * mask_width + k_col];
            }
        }
        output_data[output_row * width + output_col] = static_cast<uint8_t>(min(max(rgb, 0.0f), 255.0f));
    }
}

void test_gpu(const Image &image, const int mask_width, vector<float> &gpu_times, const bool tiling,
              uint8_t *output_data_gpu) {
    int dim = TILE_WIDTH + mask_width - 1;
    int total_bytes = dim * dim * sizeof(uint8_t);


    uint8_t *device_input;
    uint8_t *host_input;
    uint8_t *device_output;
    uint8_t *host_output;


    cudaError_t err = cudaMallocHost(&host_input, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    memcpy(host_input, image.data, image.size * sizeof(uint8_t));

    err = cudaMallocHost(&host_output, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));

    err = cudaMalloc(&device_input, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));

    err = cudaMalloc(&device_output, sizeof(uint8_t) * image.size);
    if (err != cudaSuccess) printf(cudaGetErrorString(err));


    cudaStream_t streams[3];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    float ms = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (tiling) {
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((image.width - 1) / TILE_WIDTH + 1,
                     (image.height - 1) / TILE_WIDTH + 1, 1);
        for (int k = 0; k <= NUMBER_OF_ITERATIONS; k++) {
            if (k > 2) {
                cudaEventRecord(start);
                for (int ch = 0; ch < image.channels; ch++) {
                    const int index = ch * image.width * image.height;

                    err = cudaMemcpyAsync(device_input + index, host_input + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyHostToDevice, streams[ch]);
                    if (err != cudaSuccess) printf("H2D: ", cudaGetErrorString(err));
                    gpu_kernel_with_tiling<<<dimGrid, dimBlock, total_bytes, streams[ch]>>>(
                        device_input + index, image.width, image.height, image.channels,
                        mask_width, device_output + index);
                    err = cudaMemcpyAsync(host_output + index, device_output + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyDeviceToHost, streams[ch]);
                    if (err != cudaSuccess) printf("D2H: ", cudaGetErrorString(err));
                }
                for (int ch = 0; ch < image.channels; ch++) {
                    cudaStreamSynchronize(streams[ch]);
                }
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
                gpu_times[k - 3] = ms;
            } else {
                for (int ch = 0; ch < image.channels; ch++) {
                    const int index = ch * image.width * image.height;

                    err = cudaMemcpyAsync(device_input + index, host_input + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyHostToDevice, streams[ch]);
                    if (err != cudaSuccess) printf("H2D: ", cudaGetErrorString(err));
                    gpu_kernel_with_tiling<<<dimGrid, dimBlock, total_bytes, streams[ch]>>>(
                        device_input + index, image.width, image.height, image.channels,
                        mask_width, device_output + index);
                    err = cudaMemcpyAsync(host_output + index, device_output + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyDeviceToHost, streams[ch]);
                    if (err != cudaSuccess) printf("D2H: ", cudaGetErrorString(err));
                }
                for (int ch = 0; ch < image.channels; ch++) {
                    cudaStreamSynchronize(streams[ch]);
                }
            }
        }
    } else {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((image.width + dimBlock.x - 1) / dimBlock.x,
                     (image.height + dimBlock.y - 1) / dimBlock.y);
        for (int k = 0; k <= NUMBER_OF_ITERATIONS; k++) {
            if (k > 2) {
                cudaEventRecord(start);
                for (int ch = 0; ch < image.channels; ch++) {
                    const int index = ch * image.width * image.height;
                    err = cudaMemcpyAsync(device_input + index, host_input + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyHostToDevice, streams[ch]);
                    if (err != cudaSuccess) printf("H2D: ", cudaGetErrorString(err));
                    gpu_kernel<<<dimGrid, dimBlock, 0, streams[ch]>>>(
                        device_input + index, image.width, image.height, image.channels,
                        mask_width, device_output + index);
                    err = cudaMemcpyAsync(host_output + index, device_output + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyDeviceToHost, streams[ch]);
                    if (err != cudaSuccess) printf("D2H: ", cudaGetErrorString(err));
                }

                for (int ch = 0; ch < image.channels; ch++) {
                    cudaStreamSynchronize(streams[ch]);
                }

                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
                gpu_times[k - 3] = ms;
            } else {
                for (int ch = 0; ch < image.channels; ch++) {
                    const int index = ch * image.width * image.height;

                    err = cudaMemcpyAsync(device_input + index, host_input + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyHostToDevice, streams[ch]);
                    gpu_kernel<<<dimGrid, dimBlock, 0, streams[ch]>>>(
                        device_input + index, image.width, image.height, image.channels,
                        mask_width, device_output + index);
                    err = cudaMemcpyAsync(host_output + index, device_output + index,
                                          sizeof(uint8_t) * image.width * image.height,
                                          cudaMemcpyDeviceToHost, streams[ch]);
                    if (err != cudaSuccess) printf("D2H: ", cudaGetErrorString(err));
                }
                for (int ch = 0; ch < image.channels; ch++) {
                    cudaStreamSynchronize(streams[ch]);
                }
            }
        }
    }

    printf("Tempo di esecuzione GPU kernel %dx%d (ms): \n", mask_width, mask_width);
    printf("Min: %.4f\n", *min_element(gpu_times.begin(), gpu_times.end()));
    printf("Max: %.4f\n", *max_element(gpu_times.begin(), gpu_times.end()));
    const float avg = mean(gpu_times);
    printf("Avg: %.4f\n", avg);
    printf("Std: %.4f\n", standard_dev(gpu_times, avg));

    for (int ch = 0; ch < image.channels; ch++) {
        cudaStreamDestroy(streams[ch]);
    }

    memcpy(output_data_gpu, host_output, image.size * sizeof(uint8_t));

    cudaFreeHost(host_input);
    cudaFreeHost(host_output);
    cudaFree(device_input);
    cudaFree(device_output);
}

void test_wrapper(const Image &image, vector<float> &cpu_times, vector<float> &gpu_times, string name, float* kernel3x3, float* kernel7x7, float* kernel11x11) {

    cudaError_t err;
    string filename;

    auto* output_data_cpu = new uint8_t[image.size];
    auto* output_data_gpu = new uint8_t[image.size];

    printf("CPU...\n");

    int mask_width = 3;
    test_cpu(image, mask_width, kernel3x3, output_data_cpu, cpu_times);
    Image result3_image_cpu(image.width, image.height, image.channels, output_data_cpu);
    result3_image_cpu.data = toInterleaved(result3_image_cpu);
    filename = "result_gaussian_kernel_3X3_cpu_" + name + ".png";
    bool result = result3_image_cpu.writeImage(filename.c_str());


    mask_width = 7;
    test_cpu(image, mask_width, kernel7x7, output_data_cpu, cpu_times);
    Image result7_image_cpu(image.width, image.height, image.channels, output_data_cpu);
    result7_image_cpu.data = toInterleaved(result7_image_cpu);
    filename = "result_gaussian_kernel_7X7_cpu_" + name + ".png";
    result = result7_image_cpu.writeImage(filename.c_str());

    mask_width = 11;
    test_cpu(image, mask_width, kernel11x11, output_data_cpu, cpu_times);
    Image result11_image_cpu(image.width, image.height, image.channels, output_data_cpu);
    result11_image_cpu.data = toInterleaved(result11_image_cpu);
    filename = "result_gaussian_kernel_11X11_cpu_" + name + ".png";
    result = result11_image_cpu.writeImage(filename.c_str());


    printf("\nGPU with tiling...\n");

    mask_width = 3;
    err = cudaMemcpyToSymbol(kernel, kernel3x3, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, output_data_gpu);
    Image result3_gpu_image_tiling(image.width, image.height, image.channels, output_data_gpu);
    result3_gpu_image_tiling.data = toInterleaved(result3_gpu_image_tiling);
    filename = "result_gaussian_kernel_3X3_gpu_tiling_" + name + ".png";
    result = result3_gpu_image_tiling.writeImage(filename.c_str());


    mask_width = 7;
    err = cudaMemcpyToSymbol(kernel, kernel7x7, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, output_data_gpu);
    Image result7_gpu_image_tiling(image.width, image.height, image.channels, output_data_gpu);
    result7_gpu_image_tiling.data = toInterleaved(result7_gpu_image_tiling);
    filename = "result_gaussian_kernel_7X7_gpu_tiling_" + name + ".png";
    result = result7_gpu_image_tiling.writeImage(filename.c_str());

    mask_width = 11;
    err = cudaMemcpyToSymbol(kernel, kernel11x11, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, output_data_gpu);
    Image result11_gpu_image_tiling(image.width, image.height, image.channels, output_data_gpu);
    result11_gpu_image_tiling.data = toInterleaved(result11_gpu_image_tiling);
    filename = "result_gaussian_kernel_11X11_gpu_tiling_" + name + ".png";
    result = result11_gpu_image_tiling.writeImage(filename.c_str());


    printf("\nGPU without tiling...\n");

    mask_width = 3;
    err = cudaMemcpyToSymbol(kernel, kernel3x3, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, true, output_data_gpu);
    Image result3_gpu_image(image.width, image.height, image.channels, output_data_gpu);
    result3_gpu_image.data = toInterleaved(result3_gpu_image);
    filename = "result_gaussian_kernel_3X3_gpu_" + name + ".png";
    result = result3_gpu_image.writeImage(filename.c_str());

    mask_width = 7;

    err = cudaMemcpyToSymbol(kernel, kernel7x7, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, false, output_data_gpu);
    Image result7_gpu_image(image.width, image.height, image.channels, output_data_gpu);
    result7_gpu_image.data = toInterleaved(result7_gpu_image);
    filename = "result_gaussian_kernel_7X7_gpu_" + name + ".png";
    result = result7_gpu_image.writeImage(filename.c_str());

    mask_width = 11;
    err = cudaMemcpyToSymbol(kernel, kernel11x11, mask_width * mask_width * sizeof(float));
    if (err != cudaSuccess) printf(cudaGetErrorString(err));
    test_gpu(image, mask_width, gpu_times, false, output_data_gpu);
    Image result11_gpu_image(image.width, image.height, image.channels, output_data_gpu);
    result11_gpu_image.data = toInterleaved(result11_gpu_image);
    filename = "result_gaussian_kernel_11X11_gpu_" + name + ".png";
    result = result11_gpu_image.writeImage(filename.c_str());

}
