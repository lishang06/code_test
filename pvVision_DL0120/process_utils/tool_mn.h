#ifndef _TOOL_MN_H_
#define _TOOL_MN_H_

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

void rotationImg(cv::Mat& src, cv::Mat& dst, int angle, int orientation);
void rotationLocation(int* location, int w, int h, int diff,  int angle, int orientatio, int padding, int img_w, int img_h);
void imgPpading(cv::Mat& src, cv::Mat& dst, int angle, int orientation);
int initTrack_rotation(float* init_location, int src_w, int src_h, int dst_w, int dst_h);
void currentAngleLocation(float* location, float* result, int angle, int orient, int src_w, int src_h, int dst_w, int dst_h, int img_w, int img_h);

#endif
