//
// Created by filippo on 19/01/26.
//

#ifndef UNTITLED1_UTILS_H
#define UNTITLED1_UTILS_H
#include <cstdint>
#include <cstdio>

using namespace std;



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






#endif //UNTITLED1_UTILS_H