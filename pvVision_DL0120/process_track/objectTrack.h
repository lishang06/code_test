#pragma once
// #ifndef OBJECTTRACK_H
// #define OBJECTTRACK_H

// #ifdef __cplusplus
// #if __cplusplus
// extern "C"{
// #endif
// #endif /* __cplusplus */


#include <string>
#include <vector>
#include "pvtrack.h"
#include "opencv2/opencv.hpp"

void trackInit(float* roi, cv::Mat imgData , int imgH ,int imgW);
void trackUpdate(int FrameID, cv::Mat imgData , float* result, int factor);
void trackSetParam(int scale_fl , int pchannel_num, int schannel_num , int isPHist , int isSHist, int hogInd);

// #ifdef __cplusplus
// #if __cplusplus
// }
// #endif
// #endif 

// #endif 