//
// Created by filippo on 19/01/26.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils.h"
#include "../../library/stb_image.h"
#include "../../library/stb_image_write.h"

Image::Image(const char *filename) {
    if (readImage(filename)) {
        printf("Letta l'immagine appartenente al file %s\n", filename);
        size = width * height * channels;
    } else {
        printf("Errore nella lettura dell'immagine appartenente al file %s\n", filename);
    }
}

Image::Image(const int width, const int height, const int channels, uint8_t *data) : data(data),
    size(width * height * channels), width(width), height(height), channels(channels) {
}


bool Image::readImage(const char *filename) {
    data = stbi_load(filename, &width, &height, &channels, 0);
    return data != nullptr;
}

bool Image::writeImage(const char *filename) const {
    return stbi_write_jpg(filename, width, height, channels, data, 100) != 0;
}


float mean(const vector<float>& vector) {
    float sum = 0.0f;
    for (const float element: vector) {
        sum += element;
    }
    return sum / static_cast<float>(vector.size());
}

float standard_dev(const vector<float>& vector, const float mean) {
    float sum = 0.0f;
    for (const float element: vector) {
        sum += (element - mean) * (element - mean);
    }

    return sqrt(sum / static_cast<float>(vector.size()));
}

uint8_t *toPlanar(const Image &image) {
    uint8_t *data = new uint8_t[image.width * image.height * image.channels];
    for (int channel = 0; channel < image.channels; channel++) {
        for (int col = 0; col < image.width; col++) {
            for (int row = 0; row < image.height; row++) {
                data[(channel * image.width * image.height) + row * image.width + col] = image.data[
                    (row * image.width + col) * image.channels + channel];
            }
        }
    }
    return data;
}

uint8_t *toInterleaved(const Image &image) {
    uint8_t *data = new uint8_t[image.width * image.height * image.channels];
    for (int channel = 0; channel < image.channels; channel++) {
        for (int col = 0; col < image.width; col++) {
            for (int row = 0; row < image.height; row++) {
                data[(row * image.width + col) * image.channels + channel] = image.data[
                    (channel * image.width * image.height) + row * image.width + col];
            }
        }
    }
    return data;
}

void generateGaussianKernel(const int size, const float sigma, float *kernel) {
    memset(kernel, 0.0f, sizeof(float) * size * size);
    float sum = 0.0f;
    const int half_size = size / 2;
    const int res = 2.0f * sigma * sigma;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            const int x = i - half_size;
            const int y = j - half_size;
            kernel[i * size + j] = exp(-(x * x + y * y) / res);
            sum += kernel[i * size + j];
        }
    }

    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }

}
