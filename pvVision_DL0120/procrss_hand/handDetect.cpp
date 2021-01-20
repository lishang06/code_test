#include <algorithm>
#include <map>
#include "handDetect.h"

#include <iostream>
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/stat.h>

HandDetect::HandDetect()
{ }

HandDetect::~HandDetect()
{ }

// 1*3*128*128
void HandDetect::load(const std::string &model_path, int num_thread)
{
    std::vector<std::string> tmpp = { model_path + "/handDetect_V2.mnn" };   //faceDetector.mnn
    HandHD.load_param(tmpp, num_thread);
}


int HandDetect::handDetectIternal(cv::Mat& img_, std::vector<cv::Rect> &hands)
{
    HandHD.handDetect_infer_img(img_, hands);
    return 1;
}

int HandDetect::detect(const cv::Mat& img_, std::vector<cv::Rect> &hands)
{   
    if (img_.empty()) return 0;
    cv::Mat testImg;

    // resize and normal
    cv::resize(img_, testImg, cv::Size(176,176));  //320 240

    //testImg.convertTo(testImg, CV_32FC3);

    //testImg = (testImg - img_mean) / img_std;
    
    handDetectIternal(testImg, hands);

    return 1;
}
