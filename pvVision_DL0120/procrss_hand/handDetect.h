#pragma once

#include <string>
#include <vector>
#include "HandDetectNet.h"
#include "opencv2/opencv.hpp"

class HandDetect 
{
public:
    HandDetect();
    ~HandDetect();
	
	//算法模型初始化
    void load(const std::string& model_path,int num_thread = 1);
	
	//算法单帧处理,返回人脸质量判断的结果
    int detect(const cv::Mat& img_, std::vector<cv::Rect> &hands);

private:
	//网络 base model
    Inference_engineHD HandHD;
    int handDetectIternal(cv::Mat& img_, std::vector<cv::Rect> &hands);
};