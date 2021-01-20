#include <algorithm>
#include <map>
#include "shoulderDetect.h"

#include <iostream>
#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include <sys/stat.h>



ShoulderDetect::ShoulderDetect()
{ }

ShoulderDetect::~ShoulderDetect()
{ }

// 1*3*320*320 rgb
void ShoulderDetect::load(const std::string &model_path, int num_thread)
{
    // std::vector<std::string> tmpp = { model_path + "/personDetect_v1.mnn" };   //faceDetector.mnn MobileNetV2_YOLOv3_Nano_Person_resize_2
    //  std::vector<std::string> tmpp = { model_path + "/MobileNetV2_YOLOv3_Nano_Person_resize_nearest.mnn" };    //personDetect_v1_826  MobileNetV2_YOLOv3_Nano_Person_resize_nearest
//    std::vector<std::string> tmpp = { model_path + "/yolo_fast_should_1029.mnn" };
    //std::vector<std::string> tmpp = { model_path + "/yolo-fastest_mn.mnn" };  //head_shoulder_yolo-fastest.mnn
    std::vector<std::string> tmpp = { model_path + "/head_shoulder_yolo-fastest.mnn" };  //head_shoulder_yolo-fastest.mnn
    ShoulderD.load_param(tmpp, num_thread);
}


int ShoulderDetect::shoulderDetectIternal(const char* path, cv::Mat& img_, std::vector<cv::Rect> &shoulders, int flag)
{
    ShoulderD.shoulderDetect_infer_img(path, img_, shoulders, flag);
    return 1;
}

int ShoulderDetect::detect(const char* path, const cv::Mat& img_, std::vector<cv::Rect> &shoulders, int flag)
{   
    if (img_.empty()) return 0;
    cv::Mat testImg;
    
    if( flag==1 )
    {
        cv::resize(img_, testImg, cv::Size(320,256));  //320 240
    }
    else if( flag==2 )
    {
        cv::resize(img_, testImg, cv::Size(160,128));  //320 240
    }
    
    
    
    shoulderDetectIternal(path, testImg, shoulders, flag);


    return 1;
}
