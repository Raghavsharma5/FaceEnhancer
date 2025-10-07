#include "image_processor.h"
#include <opencv2/opencv.hpp>
#include <iostream>

std::vector<cv::Rect> ImageProcessor::detectFaces(const cv::Mat& grayImage) {
    cv::CascadeClassifier faceCascade;
    // Load the classifier from the models folder
    if (!faceCascade.load("../models/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load face cascade classifier. Make sure the file is in ../models/" << std::endl;
        return std::vector<cv::Rect>();
    }

    std::vector<cv::Rect> faces;
    // Detect faces with some tuned parameters
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    return faces;
}

cv::Mat ImageProcessor::denoiseImage(const cv::Mat& image) {
    cv::Mat denoised;
    // Non-local means denoising is very effective for preserving detail
    fastNlMeansDenoisingColored(image, denoised, 10, 10, 7, 21);
    return denoised;
}

cv::Mat ImageProcessor::enhanceContrast(const cv::Mat& image) {
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split into L, A, B channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    // Apply CLAHE to the L (Lightness) channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(labChannels[0], labChannels[0]);

    // Merge the channels back
    cv::merge(labChannels, labImage);

    // Convert back to BGR color space
    cv::Mat result;
    cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);
    return result;
}

cv::Mat ImageProcessor::sharpenImage(const cv::Mat& image) {
    cv::Mat sharpened;
    // A simple but effective sharpening kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(image, sharpened, -1, kernel);
    return sharpened;
}