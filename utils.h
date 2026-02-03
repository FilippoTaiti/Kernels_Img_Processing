//
// Created by filippo on 19/01/26.
//

#ifndef UNTITLED1_UTILS_H
#define UNTITLED1_UTILS_H
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
#define NUMBER_OF_ITERATIONS 102



struct Image {
    uint8_t* data = nullptr;
    size_t size = 0;

    int width{};
    int height{};
    int channels{};

    explicit Image(const char* filename);
    Image(int width, int height, int channels, uint8_t* data); // Costruttore che permette di creare un'immagine

    bool readImage(const char* filename);
    bool writeImage(const char* filename) const;
};

Image grayscale(Image& image);

float mean(const vector<float>& vector);
float standard_dev(const vector<float>& vector, const float mean);

uint8_t* toPlanar(const Image& image);
uint8_t* toInterleaved(const Image& image);

void generateGaussianKernel(int size, float sigma, float* kernel);




#endif //UNTITLED1_UTILS_H