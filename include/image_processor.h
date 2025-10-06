#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessor {
public:
    static cv::Mat resizeImage(const cv::Mat& image, int width, int height);
    static cv::Mat normalizeImage(const cv::Mat& image);
    static std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    static cv::Mat extractFace(const cv::Mat& image, const cv::Rect& faceRect);
    static cv::Mat sharpenImage(const cv::Mat& image);
    static cv::Mat denoiseImage(const cv::Mat& image);
    static cv::Mat adjustBrightnessContrast(const cv::Mat& image, double alpha, int beta);
};

#endif // IMAGE_PROCESSOR_H