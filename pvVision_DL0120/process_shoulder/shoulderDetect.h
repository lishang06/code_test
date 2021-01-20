#pragma once

#include <string>
#include <vector>
#include "ShoulderDetectNet.h"
#include "opencv2/opencv.hpp"

class ShoulderDetect 
{
public:
    ShoulderDetect();
    ~ShoulderDetect();
	
	//算法模型初始化
    void load(const std::string& model_path,int num_thread = 1);
	
	//算法单帧处理,返回人脸质量判断的结果
    int detect(const char* path, const cv::Mat& img_, std::vector<cv::Rect> &shoulders, int flag);

private:
	//网络 base model
    Inference_engineSD ShoulderD;
    int shoulderDetectIternal(const char* path, cv::Mat& img_, std::vector<cv::Rect> &shoulders, int flag);
};
