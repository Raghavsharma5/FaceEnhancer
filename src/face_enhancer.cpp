#include "face_enhancer.h"
#include <iostream>

FaceEnhancer::FaceEnhancer() {
    // Constructor is empty, initialization is done in initialize()
}

bool FaceEnhancer::initialize() {
    // Load the face detection cascade
    if (!faceCascade.load("../models/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load face cascade classifier." << std::endl;
        return false;
    }
    return true;
}

cv::Mat FaceEnhancer::enhanceFaceRegion(const cv::Mat& faceRegion) {
    // Step 1: Denoise the face
    cv::Mat denoisedFace = ImageProcessor::denoiseImage(faceRegion);

    // Step 2: Enhance contrast
    cv::Mat contrastFace = ImageProcessor::enhanceContrast(denoisedFace);

    // Step 3: Sharpen the details
    cv::Mat sharpenedFace = ImageProcessor::sharpenImage(contrastFace);

    // Step 4: Blend with the original to maintain a natural look
    cv::Mat finalFace;
    cv::addWeighted(faceRegion, 0.3, sharpenedFace, 0.7, 0, finalFace);

    return finalFace;
}

bool FaceEnhancer::processImage(const std::string& inputPath, const std::string& outputPath) {
    // Load the input image
    cv::Mat inputImage = cv::imread(inputPath);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image at " << inputPath << std::endl;
        return false;
    }

    // Create a copy for the output
    cv::Mat outputImage = inputImage.clone();

    // Convert to grayscale for face detection
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Detect faces
    std::vector<cv::Rect> faces = ImageProcessor::detectFaces(grayImage);

    if (faces.empty()) {
        std::cout << "No faces detected in the image. Saving original." << std::endl;
        cv::imwrite(outputPath, outputImage);
        return true;
    }

    std::cout << "Detected " << faces.size() << " face(s). Enhancing..." << std::endl;

    // Process each detected face
    for (const cv::Rect& face : faces) {
        // Add some padding around the face to avoid harsh borders
        int padding = 25;
        cv::Rect paddedFace(
            std::max(0, face.x - padding),
            std::max(0, face.y - padding),
            std::min(inputImage.cols - face.x + padding, face.width + 2 * padding),
            std::min(inputImage.rows - face.y + padding, face.height + 2 * padding)
        );

        // Extract the face region
        cv::Mat faceRegion = inputImage(paddedFace);

        // Enhance the face
        cv::Mat enhancedFace = enhanceFaceRegion(faceRegion);

        // Put the enhanced face back into the output image
        enhancedFace.copyTo(outputImage(paddedFace));
    }

    // Save the result
    if (!cv::imwrite(outputPath, outputImage)) {
        std::cerr << "Error: Could not save the enhanced image to " << outputPath << std::endl;
        return false;
    }

    return true;
}