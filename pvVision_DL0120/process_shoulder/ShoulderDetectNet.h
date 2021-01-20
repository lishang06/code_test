#ifndef _SHOULDERDETECTNET_H_
#define _SHOULDERDETECTNET_H_

#include <vector>
#include <string>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>

#include "opencv2/opencv.hpp"
/*
	获取网络输出, 保存成cv::Mat
*/
class Inference_engineSD
{
public:
    Inference_engineSD();
    ~Inference_engineSD();
	
	// 加载网络参数 Session初始化
    int load_param(std::vector<std::string> &file, int num_thread = 4);

	// 网络infer, 输入cv::Mat 输出vector<Mat>, 支持多输出
    int shoulderDetect_infer_img(const char* path, cv::Mat& img, std::vector<cv::Rect> &shoulders, int flag);

private: 
    MNN::Interpreter* netPtr_shoulderDetect;
	MNN::Session* sessionPtr_shoulderDetect;
    MNN::CV::ImageProcess::Config config;
};

typedef struct shoulderRect{
    int x;
    int y;
    int width;
    int height;
    float score;
};

#define mnnSD_sigmoid(x) (float)(1.0f / (1+exp(-x)) )
#endif
