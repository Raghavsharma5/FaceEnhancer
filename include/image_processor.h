#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

// A utility class for common image processing operations
class ImageProcessor {
public:
    // Detects faces in a grayscale image
    static std::vector<cv::Rect> detectFaces(const cv::Mat& grayImage);

    // Applies a denoising filter to a color image
    static cv::Mat denoiseImage(const cv::Mat& image);

    // Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    static cv::Mat enhanceContrast(const cv::Mat& image);

    // Applies a sharpening kernel to the image
    static cv::Mat sharpenImage(const cv::Mat& image);
};

#endif // IMAGE_PROCESSOR_H