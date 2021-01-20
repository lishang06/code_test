#include "objectTrack.h"


//using namespace cv;

static PVTrack pvTrack;
//// 跟踪模板初始化
//void trackInit(float* roi, cv::Mat imgData , int imgH ,int imgW)
//{
//    unsigned char* frameBGR = (unsigned char*)malloc(640*480*4);
//    cv::Mat grayImage;
//    cvtColor(imgData, grayImage, cv::COLOR_RGB2GRAY);
//
//    // 图像格式的转换
//    vector<cv::Mat> channels;
//    split(imgData,channels);
//    cv::Mat B = channels.at(2);
//    cv::Mat G = channels.at(1);
//    cv::Mat R = channels.at(0);
//
//    int w = B.cols;  //640
//    int h = B.rows;  //480
//
////    LOGD(" w ,h is %d, %d ", imgW, imgH);
////    LOGD(" box is %f, %f, %f, %f ", roi[0], roi[1], roi[2], roi[3]);
//
//    memcpy(frameBGR,B.data,sizeof(unsigned char)*w*h);
//    memcpy((frameBGR+(w*h)),G.data,sizeof(unsigned char)*w*h);
//    memcpy((frameBGR+2*(w*h)),R.data,sizeof(unsigned char)*w*h);
//    memcpy((frameBGR+3*(w*h)),grayImage.data,sizeof(unsigned char)*w*h);
//
//    // 跟踪初始化
//    pvTrack.init(roi,frameBGR,imgH,imgW);
//    
//
//    free(frameBGR);   
//}

// 跟踪模板更新
void trackUpdate(int FrameID, cv::Mat imgData , float* result, int factor)
{   
    unsigned char* frameBGR = (unsigned char*)malloc(640*480*4);
    cv::Mat grayImage;
    cvtColor(imgData, grayImage, cv::COLOR_BGR2GRAY);

    // 图像格式的转换
    vector<cv::Mat> channels;
    split(imgData,channels);
    cv::Mat B = channels.at(0);
    cv::Mat G = channels.at(1);
    cv::Mat R = channels.at(2);

    int w = B.cols;  //640
    int h = B.rows;  //480
    memcpy(frameBGR,B.data,sizeof(unsigned char)*w*h);
    memcpy((frameBGR+(w*h)),G.data,sizeof(unsigned char)*w*h);
    memcpy((frameBGR+2*(w*h)),R.data,sizeof(unsigned char)*w*h);
    memcpy((frameBGR+3*(w*h)),grayImage.data,sizeof(unsigned char)*w*h);

    // 跟踪模板更新
    pvTrack.update(FrameID,frameBGR,result); 

    free(frameBGR);   
}

// 跟踪参数的设置，包含尺度信息
void trackSetParam(int scale_fl , int pchannel_num, int schannel_num , int isPHist , int isSHist, int hogInd)
{
    pvTrack.setParam(scale_fl, pchannel_num, schannel_num, isPHist, isSHist, hogInd);
}


