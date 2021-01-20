#ifndef TRACKTOOL_H
#define TRACKTOOL_H

#include <stdio.h>
#include <float.h>
#include <string>
#include <cmath>
#include "sMatrix.h"
extern float tableFeature[327680];

void createPositionHanningMats(int w , int h , int d , float* data);
void createScaleHanningMats(int len , float* data);
void createPosGaussianPeak(int sizey, int sizex , float sigmaFac ,fftwf_complex* resData);
void createScaleGaussianPeak(int length, float scale_sigma , fftwf_complex* resData);
void  mulSpectrums(fftwf_complex * x1, fftwf_complex *  x2,fftwf_complex *r,  int w,int h,bool conj );
void complexMultiplication(fftwf_complex* a, fftwf_complex* b, fftwf_complex* r,int w,int h);
void complexDivision(fftwf_complex* a, fftwf_complex*  b, fftwf_complex * r, int w,int h);
void rearrange( float*img,int cols,int rows);
void rearrangeAll( float*img,int cols,int rows);

void maxLoc(float*z,  int width, int height, sPoint2i *pi,   float *pv );
int maxLocOptimal(float*z,  int width, int height, sPoint2i *pi,   float *pv );
float subPixelPeak(float left, float center, float right);
int fixRect(sRect r , int w , int h);
sRect getScaleRect(sRect r , float factor);
sMatc cutRect(unsigned char* imgData , int imgH, int imgW,  const sRect  roi);
void cutRectSimple(unsigned char* imgData , unsigned char* resData, int imgH, int imgW,  const sRect roi);
void cutRectInRect(sRect baseRoi , sRect nowRoi, unsigned char* baseData, unsigned char* resData);
void resize(unsigned char* srcData , int cols , int rows , float* resData, int w1,int h1);
void avgBlur(unsigned char * srcData ,int h, int w,  unsigned char* dstData, int kSize);
int isValidRect(sRect r , int imgH, int imgW);

void preDealRoi(float* roi,  int maxP);
float getPreciseScaleFactor(float y , float y1 , float y2, int x , float a);

void save2txt(float* data, int rows , int cols, const char*name);
void save2txtcp(fftwf_complex*data,int h,int w,const char*name);

float iouTrackTool(float* box0, float* box1,  int flag);

#endif // TRACKTOOL_H
