#ifndef _HANDRECOGNITIONNET_H_
#define _HANDRECOGNITIONNET_H_

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
class Inference_engineHR
{
public:
    Inference_engineHR();
    ~Inference_engineHR();
	
	// 加载网络参数 Session初始化
    int load_param(std::vector<std::string> &file, int num_thread = 4);

	// 网络infer, 输入cv::Mat 输出vector<Mat>, 支持多输出
    int HandRecognition(cv::Mat& img);

private: 
    MNN::Interpreter* netPtr_HandRecognition;
	MNN::Session* sessionPtr_HandRecognition;
    MNN::CV::ImageProcess::Config config;
};
#endif
