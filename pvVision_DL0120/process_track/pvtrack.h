#ifndef PVTRACK_H
#define PVTRACK_H

#include <string>
#include <stdio.h>
#include "fftw3.h"
#include <float.h>

#include "fhog.h"
#include "tracktool.h"
//#include "./tool/tablefeature.h"

using namespace std;

#define getFactor(x) pow(sStep,(((double)(sNscales-1)/2.0)-x))
#define getFactorIndex(x) (((sNscales-1)/2)-x)

class PVTrack
{
//method
public:
    PVTrack();
    ~PVTrack();
    void initData();
    void setParam(int scale_fl , int pchannel_num, int schannel_num , int isPHist , int isSHist, int hogInd);
    void init(float* roi, unsigned char* imgData , int imgH ,int imgW, int initflag);
    void update(int FrameID, unsigned char* imgData , float* result);
    void updateTemplateP(int FrameID, unsigned char* imgData , float* result, int detectFlag);
    void update5S(int FrameID, unsigned char* imgData , int imgH , int imgW, float* result);
private:
    void pTrain(float* sfeature, float train_interp_factor);
    void sTrain(sRect rect, unsigned char* imgData , int imgH , int imgW,  bool init, float train_interp_factor);
    sPoint2f pDetect(float* zdata , float* xdata,float* maxV);
    float sDetect(sRect rect, unsigned char* imgData , int imgH, int imgW);
    void initTmplSz(sRect r, int initflag);
    void deinitTmplSz();
    void getPFeatures(sRect r, unsigned char* imgData , int imgH, int imgW,float* featureD);
    void getSFeatures(sRect rect, unsigned char* imgData , int imgH , int imgW, fftwf_complex* dstFeature);
    void getSFeaturesInTrain(int factorIndex, sRect rect, unsigned char* imgData , int imgH , int imgW, fftwf_complex* dstFeature);
    void getEstimateFeatures(sRect r, unsigned char* imgData , int imgH, int imgW,float* featureD);
    void gaussianCorrelation(float* f1data , float* f2data, float* resData, int width , int height, int depth, float sigma);

    void getEstimateScore();
//properties
public:
    sRect exRoi;
    int reliable;
private:
    int HEIGHT;
    int WIDTH;
    int scaleFlag;
    int pNChannels;
    int sNChannels;
    int isPHistogram;
    int isSHistogram;
    int histLen;
    int hogIndex;

    int effectFlag;

    float padding;
    int cell_size;
    float CUR_SCALE;

    sRect initRoi;
    sRect initPRoi;
    /*********************************   position   *********************************/
    sMatf pX;      float* pXBuf;  // position feature
    sMatcf pYcp;   fftwf_complex* pYBuf;      //position result complex
    sMatcf pAcp;   fftwf_complex* pABuf;      //position alphaf complex
    sMatf  pHann;  float* pHannBuf;
    sMatf pK;      float* pKBuf;  // position kernel
    sMatf pXTmp;      float* pXTmpBuf;  // position feature temple
    sMatcf pAcpTmp;   fftwf_complex* pATmpBuf;      //position alphaf complex temple
    sMatf pYTmp;      float* pYTmpBuf;
    sMatcf pYcpTmp;   fftwf_complex* pYcpTmpBuf;
    sRect pRoi;     //position roi
    sSize pTmplSZ;

//    sFftwf pFftTool;


    float pLamda;      //caculate alphaf y/(k+lamda)
    float pYSigma;      //gaussian output param
    float pYita;         //learn rate
    float pKSigma;        //kernel sigma
    int pTmplLen;          //position template window size
    int pTmplLenShort;
    float pSzFactorH;
    float pSzFactorW;
    int pPatchSZ[3];



    fftwf_plan pfft2_plan,pifft2_plan;
    fftwf_complex * pfft_input;
    fftwf_complex * pfft_output;
    fftwf_complex * ptmp;
    fftwf_complex * pifft_input ;
    fftwf_complex * pifft_output;


    /*********************************   scale   *********************************/
    sMatf sX;       float* sXBuf;
    sMatcf sXFft;   fftwf_complex* sXFftBuf;
    sMatcf  sHNumcp;  fftwf_complex* sHNumBuf;
    sMatf  sHDen;     float* sHDenBuf;
    sMatcf sYcp;      fftwf_complex* sYBuf;
    sMatcf  sHNumcpTmp;  fftwf_complex* sHNumTmpBuf;
    sMatf  sHDenTmp;     float* sHDenTmpBuf;
    sRect sRoi;         //scale roi
    sSize sTmplSZ;

    int sNscales;
    float sStep;
    float sLamda;
    float sSigma;
    int sTmplLen;


    float sYita;

    int sPatchSZ[3];
    float sHann[33]; // = {0,0.0096074,0.0380602,0.0842652,0.1464466,0.2222149,0.3086583,0.4024549,0.5000000,0.5975451,0.6913417,0.7777851,0.8535534,0.9157348,0.9619398,0.9903926,1.0000,0.9903926,0.9619398,0.9157348,0.8535534,0.7777851,0.6913417,0.5975451,0.5000000,0.4024549,0.3086583,0.2222149,0.1464466,0.0842652,0.0380602,0.0096074,0};

    fftwf_plan sfft_plan,sifft_plan;
    fftwf_complex * sfft_input;
    fftwf_complex * sfft_output;
    fftwf_complex * sifft_input ;
    fftwf_complex * sifft_output;


    float* sDetectFHogCache;
    fftwf_complex* sDetectFHogFFTCache;
    int sDetectFHogIndex;

    struct timespec time1={0,0};
    struct timespec time2={0,0};


    /*********************************************/
    float* eXTmpl;
    float* eXNow;

};


#endif // PVTRACK_H
