//
// Created by filippo on 19/01/26.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils.h"
#include "stb_image.h"
#include "stb_image_write.h"

Image::Image(const char *filename) {
    if (readImage(filename)) {
        printf("Letta l'immagine appartenente al file %s\n", filename);
        size = width * height * channels;
    }else {
        printf("Errore nella lettura dell'immagine appartenente al file %s\n", filename);
    }
}
Image::Image(const int width, const int height, const int channels, uint8_t* data) : data(data), size(width * height * channels), width(width) , height(height), channels(channels){}


bool Image::readImage(const char *filename) {
    data = stbi_load(filename, &width, &height, &channels, 0);
    return data != nullptr;
}

bool Image::writeImage(const char *filename) const {
    return stbi_write_jpg(filename,width,height,channels,data,100) != 0;
}

Image grayscale(Image& image) {
    if (image.channels == 1) {
        return image;
    }
    auto *data = new uint8_t[image.width*image.height];


    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            uint8_t r, g, b;
            r = image.data[(y*image.width+x)*image.channels];
            g = image.data[(y*image.width+x)*image.channels+1];
            b = image.data[(y*image.width+x)*image.channels+2];

            data[y*image.width+x] = 0.21f * r + 0.71f * g + 0.07f * b;
        }
    }
    Image output(image.width, image.height, 1, data);
    return output;

}
