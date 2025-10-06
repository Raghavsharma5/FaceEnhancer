#ifndef FACE_ENHANCER_H
#define FACE_ENHANCER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceEnhancer {
public:
    FaceEnhancer();
    ~FaceEnhancer();
    
    bool initialize();
    cv::Mat enhanceFace(const cv::Mat& inputImage);
    bool processImage(const std::string& inputPath, const std::string& outputPath);
    
private:
    cv::Mat preprocessImage(const cv::Mat& input);
    cv::Mat applySuperResolution(const cv::Mat& face);
    cv::Mat postprocessImage(const cv::Mat& enhanced);
    
    bool loadModels();
    
    // Model paths
    std::string srModelPath;
    std::string faceDetectionModelPath;
    
    // Model variables
    void* srModel;
    void* faceDetectionModel;
};

#endif // FACE_ENHANCER_H