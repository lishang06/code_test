#include <algorithm>
#include <map>
#include "handRecognition.h"

#include <iostream>
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/stat.h>


HandRecognition::HandRecognition()
{ }

HandRecognition::~HandRecognition()
{ }

// 1*3*128*128
void HandRecognition::load(const std::string &model_path, int num_thread)
{
    // std::vector<std::string> tmpp = { model_path + "/handRecognition_v15.mnn" };   //faceDetector.mnn
//    std::vector<std::string> tmpp = { model_path + "/handRecognition_V2.mnn" };   //faceDetector.mnn
    //std::vector<std::string> tmpp = { model_path + "/handRecognition_v3.mnn" };   //faceDetector.mnn handRecognition
    std::vector<std::string> tmpp = { model_path + "/handRecognition.mnn" };   //faceDetector.mnn handRecognition
    HandHR.load_param(tmpp, num_thread);
}


int HandRecognition::handRecognitionIternal(cv::Mat& img_)
{
    int handFlag = HandHR.HandRecognition(img_);
    return handFlag;
}

int HandRecognition::recognition(const cv::Mat& img_)
{   
    if (img_.empty()) return 0;
    cv::Mat testImg;

    // resize and normal
    // cv::resize(img_, testImg, cv::Size(128,128));  //320 240
    cv::resize(img_, testImg, cv::Size(112,112));  //320 240

    // 均值文件预处理
//    uchar* imgTmp = img_.data;
//    // for(int i=0; i<128*128*3; i++)
//    for(int i=0; i<112*112*3; i++)
//    {
//        imgTmp[i] = imgTmp[i] - meanFile[i];
//    }

    //testImg.convertTo(testImg, CV_32FC3);
    //testImg = (testImg - img_mean) / img_std;
    
    int handFlag = handRecognitionIternal(testImg);

    return handFlag;
}
