#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_enhancer.h"

int main(int argc, char** argv) {
    // Check for correct command-line arguments
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image_path> <output_image_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    // Create the FaceEnhancer object
    FaceEnhancer enhancer;

    // Process the image
    if (enhancer.processImage(inputPath, outputPath)) {
        std::cout << "Successfully processed image." << std::endl;
        std::cout << "Input: " << inputPath << std::endl;
        std::cout << "Output: " << outputPath << std::endl;
    } else {
        std::cerr << "Failed to process image." << std::endl;
        return -1;
    }

    return 0;
}