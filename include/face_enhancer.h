#ifndef FACE_ENHANCER_H
#define FACE_ENHANCER_H

#include <opencv2/opencv.hpp>
#include <string>
#include "image_processor.h"

// Main class to handle the face enhancement workflow
class FaceEnhancer {
public:
    // Constructor
    FaceEnhancer();

    // Processes an image to enhance detected faces
    // Returns true on success, false on failure
    bool processImage(const std::string& inputPath, const std::string& outputPath);

private:
    cv::CascadeClassifier faceCascade;

    // Initializes the face detection model
    bool initialize();

    // Enhances a single face region
    cv::Mat enhanceFaceRegion(const cv::Mat& faceRegion);
};

#endif // FACE_ENHANCER_H