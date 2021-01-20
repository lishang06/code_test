#ifndef _HANDDETECTNET_H_
#define _HANDDETECTNET_H_

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
class Inference_engineHD
{
public:
    Inference_engineHD();
    ~Inference_engineHD();
	
	// 加载网络参数 Session初始化
    int load_param(std::vector<std::string> &file, int num_thread = 4);

	// 网络infer, 输入cv::Mat 输出vector<Mat>, 支持多输出
    int handDetect_infer_img(cv::Mat& img, std::vector<cv::Rect> &hands);

private: 
    MNN::Interpreter* netPtr_handDetect;
	MNN::Session* sessionPtr_handDetect;
    MNN::CV::ImageProcess::Config config;
};

typedef struct handRect{
    int x;
    int y;
    int width;
    int height;
    float score;
};

#define mnn_sigmoid(x) (float)(1.0f / (1+exp(-x)) )
#endif
