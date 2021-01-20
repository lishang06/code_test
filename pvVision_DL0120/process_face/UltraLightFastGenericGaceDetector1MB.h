#pragma once

#include <string>
#include <vector>
#include "UltraFacenet.h"
#include "opencv2/opencv.hpp"

class UltraLightFastGenericGaceDetector1MB 
{
public:
    UltraLightFastGenericGaceDetector1MB();
    ~UltraLightFastGenericGaceDetector1MB();
	
	//算法模型初始化
    void load(const std::string& model_path,int num_thread = 1);
	
	//算法单帧处理
    void detect(const cv::Mat& img_, std::vector<cv::Rect> &faces, 
    std::vector<landmarkFace> &landmarkBoxResult, int detectState);

private:
	//网络 base model
    Inference_engine net;
    void detectInternal(cv::Mat& img_, std::vector<cv::Rect> &faces,  std::vector<landmarkFace> &landmarkBoxResult, int detectState);

    std::vector<cv::Rect> faces;
    cv::Mat img;
    float conf_threshold = 0.6;
    float nms_threshold = 0.5;
    int OUTPUT_NUM = 4420;
    int OUTPUT_NUM_160 = 1118;
    int OUTPUT_NUM_80 = 298;
    float img_mean = 127.0;
    float img_std = 1.0;  //128.0
    float center_variance = 0.1;
    float size_variance = 0.2;
    
};
