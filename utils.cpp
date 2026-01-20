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
    data = stbi_load(filename, &width, &height, &channels, 3);
    return data != nullptr;
}

bool Image::writeImage(const char *filename) const {
    return stbi_write_jpg(filename,width,height,channels,data,100) != 0;
}
