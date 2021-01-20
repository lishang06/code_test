#include <algorithm>
#include <map>
#include "UltraLightFastGenericGaceDetector1MB.h"
#include "faceLUT.h"

#include <iostream>
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/stat.h>

UltraLightFastGenericGaceDetector1MB::UltraLightFastGenericGaceDetector1MB()
{ }

UltraLightFastGenericGaceDetector1MB::~UltraLightFastGenericGaceDetector1MB()
{ }

void UltraLightFastGenericGaceDetector1MB::load(const std::string &model_path, int num_thread)
{
//    std::vector<std::string> tmpp = { model_path + "/faceDetector_V2.mnn" };  //RFB_v1_dev255_BGR
    std::vector<std::string> tmpp = { model_path + "/RFB_v1_dev255_BGR.mnn" };  //RFB_v1_dev255_BGR
    //std::vector<std::string> tmpp = { model_path + "/faceDetector.mnn" };   
    net.load_param(tmpp, num_thread);
}


void UltraLightFastGenericGaceDetector1MB::detectInternal(cv::Mat& img_, std::vector<cv::Rect> &faces,  std::vector<landmarkFace> &landmarkBoxResult, int detectState)
{
    faces.clear();
    img = img_;
    if( detectState==1 )
    {
        net.Ultra_infer_img(img,conf_threshold, nms_threshold,OUTPUT_NUM, center_variance, size_variance, anchors,faces, landmarkBoxResult);
    }else if( detectState==2 )
    {
        net.Ultra_infer_img_160(img,conf_threshold, nms_threshold,OUTPUT_NUM_160, center_variance, size_variance, anchors_160,faces, landmarkBoxResult);
    }else
    {
        net.Ultra_infer_img_80(img,conf_threshold, nms_threshold,OUTPUT_NUM_80, center_variance, size_variance, anchors_80,faces, landmarkBoxResult);
    }
    
    return;
}

void UltraLightFastGenericGaceDetector1MB::detect(const cv::Mat& img_, std::vector<cv::Rect> &faces,  std::vector<landmarkFace> &landmarkBoxResult, int detectState)
{   
    if (img_.empty()) return;
    cv::Mat testImg;
    
    // resize and normal
    if( detectState==1 )
    {
        cv::resize(img_, testImg, cv::Size(320,240));  //320 240
    }else if( detectState==2 )
    {
        cv::resize(img_, testImg, cv::Size(160,120));  //320 240
    }else{
        cv::resize(img_, testImg, cv::Size(80,60));  //320 240
    }
    
    // detect face
    detectInternal(testImg, faces, landmarkBoxResult, detectState);

    return;
}
