#include "pvtrack.h"


PVTrack::PVTrack(){
}

PVTrack::~PVTrack(){}

void PVTrack::initData(){
    padding = 2.5;//2.5;
    cell_size = 4;
    CUR_SCALE=1.0;

    pLamda = 0.0001;      //caculate alphaf y/(k+lamda)
    pYSigma = 0.1;      //gaussian output param
    pYita = 0.02; //0.02         //learn rate
    pKSigma = 0.5;        //kernel sigma
    pTmplLen = 56;//96;          //position template window size
    pTmplLenShort = 56;//96;

    sTmplLen = 40;

    pSzFactorW = 0.0;
    pSzFactorH = 0.0;

    sNscales = 33;
    sStep = 1.02;//1.02;
    sLamda = 0.01;
    sSigma = 1.4361;

    sYita =0.01;// 0.01;//0.025;

    isPHistogram = 1;  // 表示用颜色直方图
    isSHistogram = 1;
    histLen = 10;

    scaleFlag = 2;  // 1 noscale; 2 scale
    pNChannels = 1;
    sNChannels = 1;

    hogIndex = 3;//2;  // 图像总共有4个通道，标号3表示最后一个灰色图像通道

    // LOGD(" padding is %f",  (float)padding);
    // LOGD(" cell_size is %f",  (float)cell_size);
    // LOGD(" CUR_SCALE is %f",  (float)CUR_SCALE);
    // LOGD(" pLamda is %f",  (float)pLamda);

}

void PVTrack::setParam(int scale_fl , int pchannel_num, int schannel_num , int isPHist , int isSHist, int hogInd){
    scaleFlag = scale_fl;
    pNChannels = pchannel_num;
    sNChannels = schannel_num;
    isPHistogram = isPHist;
    isSHistogram = isSHist;
    hogIndex = hogInd;
}

//sRect initRoi;
void PVTrack::init(float* roi, unsigned char* imgData , int imgH ,int imgW, int initflag){
    HEIGHT = imgH;
    WIDTH = imgW;
    preDealRoi(roi, 5);
    initData();  //参数初始化，是否支持尺度和padding尺度等信息
    
    initRoi = sRect(roi[0], roi[1], roi[2], roi[3]);
    sRoi = sRect(roi[0], roi[1],roi[2],roi[3]);
    pRoi = sRect(roi[0], roi[1],roi[2],roi[3]);
    initPRoi = sRect(pRoi.x, pRoi.y, pRoi.width, pRoi.height);
    
    initTmplSz(initRoi, initflag);
    
    getPFeatures(pRoi, imgData, HEIGHT, WIDTH, pX.data); //中心加模板宽

    //save2txt(pX.data,pX.rows,pX.cols,"pX_init.txt");

    createPosGaussianPeak(pPatchSZ[0], pPatchSZ[1], pYSigma ,pYcp.data);
    createScaleGaussianPeak(sNscales, sSigma , sYcp.data);

    pTrain(pX.data,1.0); //初始学习率为1

    //save2txtcp(pAcp.data,pAcp.rows,pAcp.cols,"pAcp_init.txt");

    if(scaleFlag == 2)
    {
        sTrain(sRoi,imgData, HEIGHT, WIDTH,true,1.0);  //true表示初始帧　
    }

    getEstimateFeatures(pRoi, imgData ,HEIGHT, WIDTH,eXTmpl);  //得分估计模板特征，只有hog特征
    
    //printf("sRoi111 is %d, %d, %d %d\n", (int)sRoi.x, (int)sRoi.y, (int)sRoi.width, (int)sRoi.height);
}

void PVTrack::updateTemplateP(int FrameID, unsigned char* imgData , float* result, int detectFlag)
{
    float scaleTmp = result[2] / sRoi.width;

    /* 越界判断 */
    int updateFlag = 1;
    if( (result[0]+result[2])>1100 || (result[1]+result[3])>640 )
    {
        updateFlag = 0;
    }
    
    /* 质量估计模板的更新,有检测进行大尺度的校正，无检测依据尺度池校正 */
    // 全部进行更新
    float posYita = pYita, scaleYita = sYita;
    if( detectFlag==1 )
    {
        posYita = 0.02;  // 0.02  位置更新小，尺度更新大
        scaleYita = 0.1;  // 0.01
        
        /* 如果得分较低，且iou相差较大或者位置为包含关系，则直接重新学习初始化 */
        if( reliable<65 )
        {
            float sRoi_tmp[4];
            sRoi_tmp[0] = sRoi.x;  sRoi_tmp[1] = sRoi.y;
            sRoi_tmp[2] = sRoi.width; sRoi_tmp[3] = sRoi.height;
            /* 判断iou和位置关系 */
            float iouSingle = iouTrackTool(sRoi_tmp, result,  1);
            float iouDouble = iouTrackTool(sRoi_tmp, result,  0);
            
            /* 相当于大尺度重新进行初始化操作 */
            if( iouSingle>0.8 && iouDouble<0.4 )
            {
                posYita = 1.0;   //实际位置模版可以不进行初始化或者稍微提高学习率即可
                scaleYita = 1.0;
            }
        }
        
        if( updateFlag==1 && result[4]>10 )
        {
            int LEN = sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]*3;
            for(int i = 0;i<LEN ;++i)
            {
                // printf("eXNow[i] is %d, %d\n",i, (int)eXNow[i]);
                eXTmpl[i] = (1 - 0.05) * eXTmpl[i] + 0.05 * eXNow[i]; //跟踪得分判断模板
            }
        }
        CUR_SCALE=1.0;
        preDealRoi(result, 5);
        initRoi = sRect(result[0], result[1], result[2], result[3]);
        sRoi = sRect(result[0], result[1],result[2],result[3]);
        pRoi = sRect(result[0], result[1],result[2],result[3]);
        initPRoi = sRect(pRoi.x, pRoi.y, pRoi.width, pRoi.height);
        sRect r;
        r.x = result[0];
        r.y = result[1];
        r.width = result[2];
        r.height = result[3];
        int padded_w = r.width * padding;//    padding = 2.5;
        int padded_h = r.height * padding;

        // 位置模板归一化因子信息更新
        if(padded_w >= padded_h)
        {
            double p = (double)r.height/(double)r.width;
            padded_w = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.5*(p-1))+r.width ;
            padded_h = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.33*(p-1))+r.height ;
            pSzFactorW = padded_w/(float)pTmplLen;
            pSzFactorH = padded_h/(float)pTmplLenShort;
        }
        else
        {
            double p = (double)r.width/(double)r.height;
            padded_w = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.33*(p-1))+r.width ;
            padded_h = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.5*(p-1))+r.height ;
            pSzFactorH = padded_h / (float) pTmplLen;
            pSzFactorW = padded_w / (float) pTmplLenShort;
        }
    }
    else
    {
        posYita = 0.02;
        scaleYita = 0.01;
        int LEN = sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]*3;
        if( updateFlag==1 )
        {
            if( result[4]>70 )
            {
                for(int i = 0;i<LEN ;++i)
                {
                    eXTmpl[i] = (1 - posYita) * eXTmpl[i] + posYita * eXNow[i];
                    // printf("eXNow[i] is %d, %d\n",i, (int)eXNow[i]);
                }
            }
        }
    }
    
    //printf("posYita scaleYita is %f, %f\n",posYita, scaleYita);
    
    /* 更新学习速率，位置处学习速率不能设置为０，否则形变较大情况下会跟踪丢失 */
//    if(reliable > 70)
//    {
//        posYita = pYita;
//    }
//    else if(reliable >60)
//    {
//        posYita = pYita*0.75;
//    }
//    else if(reliable >45)
//    {
//        posYita = pYita*0.5;
//    }
//    else
//    {
//        posYita = pYita*0.25;
//    }
//
//    if(reliable > 70)
//    {
//        scaleYita = sYita;
//    }
//    else if(reliable >60)
//    {
//        scaleYita = sYita*0.75;
//    }
//    else if(reliable>50)
//    {
//        scaleYita = sYita*0.5;
//    }
//    else if(reliable>45)
//    {
//        scaleYita = sYita*0.25;
//    }
//    else
//    {
//        scaleYita = 0;
//    }
    
    /* 新学习率 */
    if(reliable > 70)
    {
        posYita = pYita;
    }
    else if(reliable >60)
    {
        posYita = pYita*0.75;
    }
    else if(reliable >50)
    {
        posYita = pYita*0.25;
    }else{
        posYita = 0;
    }

    if(reliable > 70)
    {
        scaleYita = sYita;
    }
    else if(reliable >60)
    {
        scaleYita = sYita*0.75;
    }
    else if(reliable>50)
    {
        scaleYita = sYita*0.25;
    }else{
        scaleYita = 0;
    }
    
    sRect pRoiUpdate;
    pRoiUpdate.x = result[0];
    pRoiUpdate.y = result[1];
    pRoiUpdate.width = result[2];
    pRoiUpdate.height = result[3];
    
    getPFeatures(pRoiUpdate,imgData,HEIGHT,WIDTH,pXTmp.data);
    pTrain(pXTmp.data,posYita);
    if(scaleFlag == 2){
        sTrain(pRoiUpdate,imgData,HEIGHT,WIDTH,false,scaleYita);
    }

    /* 更新位置和尺度的大小位置信息 */

    //printf("update location is %f, %f, %f, %f\n", result[0],result[1],result[2],result[3]);

    pRoi.x = result[0];
    pRoi.y = result[1];
    pRoi.width = result[2];
    pRoi.height = result[3];

    sRoi.x = result[0];
    sRoi.y = result[1];
    sRoi.width = result[2];
    sRoi.height = result[3];
    
    //printf("sRoi12 is %d, %d, %d %d\n", (int)sRoi.x, (int)sRoi.y, (int)sRoi.width, (int)sRoi.height);
    
}

void PVTrack::update(int FrameID, unsigned char* imgData , float* result)
{
    float cx = pRoi.x + pRoi.width / 2.0f;
    float cy = pRoi.y + pRoi.height / 2.0f;

    //printf("pRoi1 is %f, %f, %f, %f\n",(float)pRoi.x, (float)pRoi.y, (float)pRoi.width, (float)pRoi.height );
    getPFeatures(pRoi,imgData,HEIGHT,WIDTH,pXTmp.data);  //中心加模板宽高

    float maxV;

    sPoint2f res = pDetect(pX.data ,pXTmp.data, &maxV);

    pRoi.x = cx - pRoi.width / 2.0f + round(((float) res.x * cell_size * pSzFactorW));
    pRoi.y = cy - pRoi.height / 2.0f + round(((float) res.y * cell_size * pSzFactorH));
    
    //printf("res.x is %f\n", res.x);
    //printf("res.y is %f\n", res.y);
    
    //printf("res.x * cell_size * pSzFactorW is %f\n", res.x * cell_size * pSzFactorW);
    //printf("res.y * cell_size * pSzFactorH is %f\n", res.y * cell_size * pSzFactorH);

    //fix scale
    cx = pRoi.x + pRoi.width / 2.0f;
    cy = pRoi.y + pRoi.height / 2.0f;

    sRoi.x = cx - sRoi.width/2.0f;
    sRoi.y = cy - sRoi.height/2.0f;

    float fixS = 1;
    if(scaleFlag == 2){
        fixS = sDetect(sRoi,imgData, HEIGHT, WIDTH);
    }
    
    //printf("fixS is %f\n", fixS);
    //printf("initRoi.width is %f, %f\n",(float)initRoi.width, (float)initRoi.height);

    /* 1 */
    pSzFactorW *= fixS;
    pSzFactorH *= fixS;
    CUR_SCALE *= fixS;
    
    //printf("sRoi1 is %d, %d, %d %d\n", (int)sRoi.x, (int)sRoi.y, (int)sRoi.width, (int)sRoi.height);

    sRoi.width = CUR_SCALE*initRoi.width;
    sRoi.height = CUR_SCALE*initRoi.height;
    sRoi.x = cx - sRoi.width / 2.0f ;
    sRoi.y = cy - sRoi.height / 2.0f;
    
    //printf("CUR_SCALE cx cy is %f, %f, %f\n",CUR_SCALE, cx, cy );
    //printf("sRoi2 is %d, %d, %d %d\n", (int)sRoi.x, (int)sRoi.y, (int)sRoi.width, (int)sRoi.height);

    pRoi.width = CUR_SCALE*initPRoi.width;
    pRoi.height = CUR_SCALE*initPRoi.height;
    pRoi.x = cx - pRoi.width / 2.0f ;
    pRoi.y = cy - pRoi.height / 2.0f;

    getEstimateFeatures(pRoi, imgData ,HEIGHT, WIDTH,eXNow);
    getEstimateScore();

    //printf("score is %d\n",reliable );
    
    int valid = isValidRect(sRoi,HEIGHT,WIDTH);
    if(valid == 0){
        reliable = 0;
    }
    
    result[0] = sRoi.x;
    result[1] = sRoi.y;
    result[2] = sRoi.width;
    result[3] = sRoi.height;
    result[4] = (float)reliable;
    result[5] = FrameID;
    
    //printf("sRoi22 is %d, %d, %d %d\n", (int)sRoi.x, (int)sRoi.y, (int)sRoi.width, (int)sRoi.height);
}

// void PVTrack::update(int FrameID, unsigned char* imgData , float* result)
// {
//     struct timespec time1={0,0};
//     struct timespec time2={0,0};

// //    fixRect(pRoi,WIDTH , HEIGHT);
//     float cx = pRoi.x + pRoi.width / 2.0f;
//     float cy = pRoi.y + pRoi.height / 2.0f;

//     getPFeatures(pRoi,imgData,HEIGHT,WIDTH,pXTmp.data);


//     float maxV;

//     sPoint2f res = pDetect(pX.data ,pXTmp.data, &maxV);

// //    if(effectFlag==-1){
// //        return;
// //    }


//     pRoi.x = cx - pRoi.width / 2.0f + round(((float) res.x * cell_size * pSzFactorW));
//     pRoi.y = cy - pRoi.height / 2.0f + round(((float) res.y * cell_size * pSzFactorH));

// //    printf("update detect: (%.3f , %.3f)\n",((float) res.x * cell_size * pSzFactorW), ((float) res.y * cell_size * pSzFactorH));

// //    fixRect(pRoi,WIDTH , HEIGHT);

//     //fix scale
//     cx = pRoi.x + pRoi.width / 2.0f;
//     cy = pRoi.y + pRoi.height / 2.0f;

//     sRoi.x = cx - sRoi.width/2.0f;
//     sRoi.y = cy - sRoi.height/2.0f;

//     float fixS = 1;
//     if(scaleFlag == 2){
//         fixS = sDetect(sRoi,imgData, HEIGHT, WIDTH);
//     }

//     pSzFactorW *= fixS;
//     pSzFactorH *= fixS;

//     CUR_SCALE *= fixS;

//     sRoi.width = CUR_SCALE*initRoi.width;
//     sRoi.height = CUR_SCALE*initRoi.height;
//     sRoi.x = cx - sRoi.width / 2.0f ;
//     sRoi.y = cy - sRoi.height / 2.0f;

//     printf("sRoi.x y w h is %f, %f, %d, %d, %f\n",sRoi.x, sRoi.y, sRoi.width, sRoi.height, (float)sRoi.width/(float)sRoi.height);


//     pRoi.width = CUR_SCALE*initPRoi.width;
//     pRoi.height = CUR_SCALE*initPRoi.height;
//     pRoi.x = cx - pRoi.width / 2.0f ;
//     pRoi.y = cy - pRoi.height / 2.0f;

// //    fixRect(sRoi,WIDTH , HEIGHT);
// //    fixRect(pRoi,WIDTH , HEIGHT);
// //    int isTrain = fixRect(exRoi,WIDTH , HEIGHT);

//     float posYita = pYita, scaleYita = sYita;
// //    if(isTrain == 1){
// //        posYita = pYita*1/3;
// //        scaleYita = sYita*1/3;
// //    }else{
//         if(reliable > 70){
//             posYita = pYita;
//         }else if(reliable >50){
//             posYita = pYita*0.75;
//         }else /*if(reliable>40)*/{
//             posYita = pYita*0.5;
//         }/*else{
//             posYita = pYita*0.25;
//         }*/

//         if(reliable > 70){
//             scaleYita = sYita;
//         }else if(reliable >60){
//             scaleYita = sYita*0.75;
//         }else if(reliable>50){
//             scaleYita = sYita*0.5;
//         }else{
//             scaleYita = sYita*0.25;
//         }
// //    }

//     getPFeatures(pRoi,imgData,HEIGHT,WIDTH,pXTmp.data);
//     pTrain(pXTmp.data,posYita);
//     if(scaleFlag == 2){
//         sTrain(sRoi,imgData,HEIGHT,WIDTH,false,scaleYita);
//     }

//     getEstimateFeatures(pRoi, imgData ,HEIGHT, WIDTH,eXNow);
//     getEstimateScore();

//     int valid = isValidRect(sRoi,HEIGHT,WIDTH);
//     if(valid == 0){
//         reliable = 0;
//     }

//     result[0] = sRoi.x;
//     result[1] = sRoi.y;
//     result[2] = sRoi.width;
//     result[3] = sRoi.height;
//     result[4] = reliable;
//     result[5] = FrameID;

// //    printf("[%f , %f , %f , %f]   reliable: %f\n", result[0],result[1],result[2],result[3],result[4]);
// //    return sRoi;
// }

void PVTrack::pTrain(float* sfeature, float train_interp_factor){
    int width = pPatchSZ[0];
    int height = pPatchSZ[1];
    int size =width*height;

    gaussianCorrelation(sfeature, sfeature, pK.data , width , height, pPatchSZ[2], pKSigma);

    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            pfft_input[i *  width + j][0] = pK.data[i*width+j];
            pfft_input[i *  width + j][1] = 0;
        }
    }
    fftwf_execute(pfft2_plan);

    for(int i=0;i<size;++i){
       pfft_output[i][0]+=pLamda;
       pfft_output[i][1]+=pLamda;
    }

    complexDivision(pYcp.data, pfft_output, pAcpTmp.data, width,height);

    int s = size*pPatchSZ[2];
    
    for(int i = 0;i<s ;++i){
        pX.data[i] =    (1 - train_interp_factor) * pX.data[i] + (train_interp_factor) * sfeature[i];  // 特征模板
    }

    for(int i = 0;i<size;++i){  // 滤波器模板
        pAcp.data[i][0]= (1 - train_interp_factor) * pAcp.data[i][0] + (train_interp_factor) * pAcpTmp.data[i][0];
        pAcp.data[i][1]= (1 - train_interp_factor) * pAcp.data[i][1] + (train_interp_factor) * pAcpTmp.data[i][1];
    }
}

void PVTrack::sTrain(sRect rect, unsigned char* imgData , int imgH , int imgW,  bool init, float train_interp_factor){
    int scaleFeaWidth = sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen);

    if(init){
        getSFeatures(rect, imgData, imgH, imgW, sXFft.data);
    }else{
        if(sDetectFHogIndex == 0){
            memcpy(sXFft.data,sDetectFHogFFTCache,sNscales*scaleFeaWidth*sizeof(fftwf_complex));
        }else{
            getSFeaturesInTrain(sDetectFHogIndex,rect, imgData, imgH, imgW, sXFft.data);
        }
    }

    memset(sHDenTmp.data,0.0,sizeof(float)*sNscales);

    for(int i=0; i<sNscales ;i++){
        for(int j=0; j<scaleFeaWidth;j++){
            sHNumcpTmp.data[i*scaleFeaWidth+j][0] = (sYcp.data[i][0]*sXFft.data[i*scaleFeaWidth+j][0] + sYcp.data[i][1]*sXFft.data[i*scaleFeaWidth+j][1]);
            sHNumcpTmp.data[i*scaleFeaWidth+j][1] = (sYcp.data[i][1]*sXFft.data[i*scaleFeaWidth+j][0] - sYcp.data[i][0]*sXFft.data[i*scaleFeaWidth+j][1]);
        }
    }

    for(int i=0; i<sNscales ;i++){
        for(int j=0; j<scaleFeaWidth;j++){
            sHDenTmp.data[i] += (sXFft.data[i*scaleFeaWidth+j][0]*sXFft.data[i*scaleFeaWidth+j][0] + sXFft.data[i*scaleFeaWidth+j][1]*sXFft.data[i*scaleFeaWidth+j][1]);
        }
    }

    if(init){
        memcpy(sHNumcp.data,sHNumcpTmp.data,sizeof(fftwf_complex)*scaleFeaWidth*sNscales);
        memcpy(sHDen.data,sHDenTmp.data,sizeof(float)*sNscales);
    }else{
        for(int i=0; i<sNscales; i++){
            for(int j=0; j<scaleFeaWidth;j++){
                sHNumcp.data[i*scaleFeaWidth+j][0] = (1-train_interp_factor)*sHNumcp.data[i*scaleFeaWidth+j][0]+train_interp_factor*sHNumcpTmp.data[i*scaleFeaWidth+j][0];
                sHNumcp.data[i*scaleFeaWidth+j][1] = (1-train_interp_factor)*sHNumcp.data[i*scaleFeaWidth+j][1]+train_interp_factor*sHNumcpTmp.data[i*scaleFeaWidth+j][1];
            }
            sHDen.data[i] = (1-train_interp_factor)*sHDen.data[i]+train_interp_factor*sHDenTmp.data[i];
        }

    }
}

sPoint2f PVTrack::pDetect(float* zdata , float* xdata, float* maxV){
    int width  = pPatchSZ[0];
    int height = pPatchSZ[1];

    gaussianCorrelation(xdata,zdata, pK.data , width , height, pPatchSZ[2], pKSigma);

//    save2txt(pK.data,height,width,"pK.txt");
//    save2txtcp(pAcp.data,height,width,"pAcp.txt");

    int ind = 0;
    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            ind = i *  width + j;
            pfft_input[ind][0] = pK.data[ind];
            pfft_input[ind][1] = 0;
        }
    }
    fftwf_execute(pfft2_plan);

    int s = height*width;
    for (int i = 0; i < s; ++i){
        pfft_output[i][0]*= 1.f / (float) s;
        pfft_output[i][1]*= 1.f / (float) s;
    }

    complexMultiplication(pAcp.data, pfft_output, pYcpTmp.data,width,height);

    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            ind = i *  width + j;
            pifft_input[ind][0]=  pYcpTmp.data[ind][0];
            pifft_input[ind][1] = pYcpTmp.data[ind][1];
        }
    }

    fftwf_execute(pifft2_plan);

    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            ind = i *  width + j;
            pYTmp.data[ind] = pifft_output[ind][0];
        }
    }


//    save2txt(pYTmp.data,height,width,"pYTmp.txt");

    sPoint2i pi;
    float pv;
    maxLoc(pYTmp.data,  width,height,&pi, &pv);
//    effectFlag = maxLocOptimal(pYTmp.data,  width,height,&pi, &pv);
    (*maxV) = pv;

//    save2txt(pYTmp.data,pYTmp.rows,pYTmp.cols,"pY.txt");
//    maxLocOptimal(pYTmp.data,  width,height,&pi, &pv);

    sPoint2f p((float)pi.x, (float)pi.y);

//    printf("postion detect: (%.0f , %.0f)\n",(p.x - (width)/2), (p.y-(height) / 2));

    if(pi.x > 0 && pi.x < width-1){
        p.x += subPixelPeak(pYTmp.data[pi.y*width+ pi.x-1], pv, pYTmp.data[pi.y*width+ pi.x+1]);
    }

    if (pi.y > 0 && pi.y < height-1){
        p.y += subPixelPeak(pYTmp.data[(pi.y-1)*width+ pi.x], pv, pYTmp.data[(pi.y+1)*width+ pi.x]);
    }

    p.x -= (width) / 2;
    p.y -= (height) / 2;

//    printf("postion detect: (%.3f , %.3f)\n",p.x, p.y);
//    printf("Pos: %.3f   ",pv);
//    reliable = (int)((int)(pv*100)%100);
    return p;
}

float PVTrack::sDetect(sRect rect, unsigned char* imgData , int imgH, int imgW){
    struct timespec time1={0,0};
    struct timespec time2={0,0};

    int scaleFeaWidth = sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen);

    getSFeatures(rect, imgData, imgH, imgW, sXFft.data);

    memset(sifft_input,0.0,sizeof(fftwf_complex)*sNscales);

    for(int i=0; i<sNscales; i++){
        for(int j=0; j<scaleFeaWidth;j++){
            sifft_input[i][0] += (sXFft.data[i*scaleFeaWidth+j][0]*sHNumcp.data[i*scaleFeaWidth+j][0] - sXFft.data[i*scaleFeaWidth+j][1]*sHNumcp.data[i*scaleFeaWidth+j][1]);
            sifft_input[i][1] += (sXFft.data[i*scaleFeaWidth+j][1]*sHNumcp.data[i*scaleFeaWidth+j][0] + sXFft.data[i*scaleFeaWidth+j][0]*sHNumcp.data[i*scaleFeaWidth+j][1]);
        }
        sifft_input[i][0] /= (sHDen.data[i]+sLamda);
        sifft_input[i][1] /= (sHDen.data[i]+sLamda);
    }

    //ifft
    fftwf_execute(sifft_plan);

    int maxScaleIndex = 0;
    float maxScaleNum = -100000;

//save2txtcp(sifft_output,1,sNscales,"scaleY.txt");

    for(int j=0; j<sNscales; j++){
        if(maxScaleNum<sifft_output[j][0]){
            maxScaleNum = sifft_output[j][0];
            maxScaleIndex = j;
        }
    }
//    printf("Scale: %.3f   \n",maxScaleNum);

//    if(maxScaleIndex>18) {printf("smaller %d\n",maxScaleIndex); maxScaleIndex=18;}
//    if(maxScaleIndex<14) {printf("bigger %d\n",maxScaleIndex); maxScaleIndex=14;}

//    printf("best scale is : %d ,   %f  \n",maxScaleIndex, getFactor(maxScaleIndex));
    sDetectFHogIndex = getFactorIndex(maxScaleIndex);
    return getFactor(maxScaleIndex);
}

void PVTrack::initTmplSz(sRect r, int initflag){
    int padded_w = r.width * padding;//    padding = 2.5;
    int padded_h = r.height * padding;

    if(padded_w >= padded_h){
//        padded_w = padded_w/padding+(padding-1)*r.height;

        double p = (double)r.height/(double)r.width;
        padded_w = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.5*(p-1))+r.width ;
        padded_h = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.33*(p-1))+r.height ;

        pSzFactorW = padded_w/(float)pTmplLen;
        pSzFactorH = padded_h/(float)pTmplLenShort;
    }else{
//        padded_h = padded_h/padding +(padding-1)*r.width;

        double p = (double)r.width/(double)r.height;
        padded_w = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.33*(p-1))+r.width ;
        padded_h = (padding-1)*sqrt(r.width*r.height)*pow(2.718281828,0.5*(p-1))+r.height ;

        pSzFactorH = padded_h / (float) pTmplLen;
        pSzFactorW = padded_w / (float) pTmplLenShort;
    }

    pTmplSZ.w = padded_w / pSzFactorW;
    pTmplSZ.h = padded_h / pSzFactorH;

    // Round to cell size and also make it even
    pTmplSZ.w = (((int)(pTmplSZ.w/(2 * cell_size))) * 2 * cell_size) + cell_size*2;
    pTmplSZ.h = (((int)(pTmplSZ.h/(2 * cell_size))) * 2 * cell_size) + cell_size*2;
    
    pTmplSZ.w = 64;
    pTmplSZ.h = 64;

    pYSigma = sqrt(r.width*r.height/pSzFactorW/pSzFactorH)*pYSigma/cell_size;

//////////////////////////////////////////////////

    if(r.width > r.height){
        sTmplSZ.w = sTmplLen;
        sTmplSZ.h = r.height*sTmplLen/r.width;
        sTmplSZ.h = (((int)(sTmplSZ.h/(2*cell_size))) * 2*cell_size) + 2*cell_size;

    }else{
        sTmplSZ.h = sTmplLen;
        sTmplSZ.w = r.width*sTmplLen/r.height;
        sTmplSZ.w = (((int)(sTmplSZ.w/(2*cell_size))) * 2*cell_size) + 2*cell_size;
    }
    sTmplSZ.w = 40;
    sTmplSZ.h = 40;


    pPatchSZ[0] = pTmplSZ.w/cell_size;
    pPatchSZ[1] = pTmplSZ.h/cell_size;
    pPatchSZ[2] = 9*3+5;

    sPatchSZ[0] = sTmplSZ.w/cell_size;
    sPatchSZ[1] = sTmplSZ.h/cell_size;
    sPatchSZ[2] = 9*3;

//    sPatchSZ[0] = sTmplSZ.w;
//    sPatchSZ[1] = sTmplSZ.h;
//    sPatchSZ[2] = 1;

    if( initflag )
    {
        // init param
        pXBuf = (float*)malloc(sizeof(float)*pPatchSZ[0]*pPatchSZ[1]*(pPatchSZ[2]*pNChannels+isPHistogram*histLen));
        pABuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*pPatchSZ[0]*pPatchSZ[1]);
        pYBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*pPatchSZ[0]*pPatchSZ[1]);
        pKBuf = (float*)malloc(sizeof(float)*pPatchSZ[0]*pPatchSZ[1]);

        pXTmpBuf = (float*)malloc(sizeof(float)*pPatchSZ[0]*pPatchSZ[1]*(pPatchSZ[2]*pNChannels+isPHistogram*histLen));
        pATmpBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*pPatchSZ[0]*pPatchSZ[1]);
        pYTmpBuf = (float*)fftwf_malloc(sizeof(float)*pPatchSZ[0]*pPatchSZ[1]);
        pYcpTmpBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*pPatchSZ[0]*pPatchSZ[1]);

        memset(pABuf,0,sizeof(fftwf_complex)*pPatchSZ[0]*pPatchSZ[1]);
        memset(pXBuf,0,sizeof(float)*pPatchSZ[0]*pPatchSZ[1]*(pPatchSZ[2]*pNChannels+isPHistogram*histLen));


        pX = sMatf(pPatchSZ[0]*pPatchSZ[1],(pPatchSZ[2]*pNChannels+isPHistogram*histLen),pXBuf);
        pAcp = sMatcf(pPatchSZ[0],pPatchSZ[1],pABuf);
        pYcp = sMatcf(pPatchSZ[0],pPatchSZ[1],pYBuf);
        pK = sMatf(pPatchSZ[0],pPatchSZ[1],pKBuf);

        pXTmp = sMatf(pPatchSZ[0]*pPatchSZ[1],(pPatchSZ[2]*pNChannels+isPHistogram*histLen),pXTmpBuf);
        pAcpTmp = sMatcf(pPatchSZ[0],pPatchSZ[1],pATmpBuf);
        pYTmp = sMatf(pPatchSZ[0],pPatchSZ[1],pYTmpBuf);
        pYcpTmp = sMatcf(pPatchSZ[0],pPatchSZ[1],pYcpTmpBuf);

        pHannBuf = (float*)malloc(sizeof(float)*pPatchSZ[0]*pPatchSZ[1]*pPatchSZ[2]);
        pHann = sMatf(pPatchSZ[0]*pPatchSZ[1],pPatchSZ[2],pHannBuf);

        pfft_input  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * pPatchSZ[0]*pPatchSZ[1]);
        pfft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * pPatchSZ[0]*pPatchSZ[1]);
        pfft2_plan = fftwf_plan_dft_2d(pPatchSZ[1],pPatchSZ[0], pfft_input, pfft_output, FFTW_FORWARD, FFTW_ESTIMATE);
        ptmp  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *pPatchSZ[0]*pPatchSZ[1]);
        pifft_input   = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *pPatchSZ[0]*pPatchSZ[1]);
        pifft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *pPatchSZ[0]*pPatchSZ[1]);
        pifft2_plan =fftwf_plan_dft_2d(pPatchSZ[1],pPatchSZ[0], pifft_input, pifft_output, FFTW_BACKWARD, FFTW_ESTIMATE);


        sXBuf = (float*)malloc(sizeof(float)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);
        sXFftBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);
        sHNumBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);
        sHDenBuf = (float*)malloc(sizeof(float)*sNscales);
        sYBuf  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * sNscales);

        sHNumTmpBuf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);
        sHDenTmpBuf = (float*)malloc(sizeof(float)*sNscales);

        sX = sMatf(sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen),sNscales,sXBuf);
        sXFft = sMatcf(sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen),sNscales,sXFftBuf);
        sHNumcp = sMatcf(sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen),sNscales,sHNumBuf);
        sHDen = sMatf(sNscales,1,sHDenBuf);
        sYcp = sMatcf(sNscales,1,sYBuf);

        sHNumcpTmp = sMatcf(sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen),sNscales,sHNumTmpBuf);
        sHDenTmp = sMatf(sNscales,1,sHDenTmpBuf);


        sfft_input  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * sNscales);
        sfft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * sNscales);
        sfft_plan = fftwf_plan_dft_1d(sNscales, sfft_input, sfft_output , FFTW_FORWARD,  FFTW_ESTIMATE);

        sifft_input   = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * sNscales);
        sifft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * sNscales);
        sifft_plan = fftwf_plan_dft_1d(sNscales, sifft_input, sifft_output , FFTW_BACKWARD,  FFTW_ESTIMATE);

        /******************* Estimate ***********************/
        eXTmpl = (float*)malloc(sizeof(float)*sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]*3);
        eXNow = (float*)malloc(sizeof(float)*sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]*3);
        /**************** end  Estimate *********************/

        sDetectFHogCache = (float*)malloc(sizeof(float)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);
        sDetectFHogFFTCache = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen)*sNscales);


        createPositionHanningMats(pPatchSZ[0],pPatchSZ[1],pPatchSZ[2],pHann.data);
        createScaleHanningMats(sNscales, sHann);
    }
}

/* 释放初始化内存 */
void PVTrack::deinitTmplSz()
{
    free(pXBuf);
    fftwf_free(pABuf);
    fftwf_free(pYBuf);
    free(pKBuf);
    
    free(pXTmpBuf);
    fftwf_free(pATmpBuf);
    free(pYTmpBuf);
    fftwf_free(pYcpTmpBuf);

    free(pHannBuf);

    fftwf_free(pfft_input);
    fftwf_free(pfft_output);
    fftwf_free(pfft2_plan);
    fftwf_free(ptmp);
    fftwf_free(pifft_input);
    fftwf_free(pifft_output);
    fftwf_free(pifft2_plan);
  
    free(sXBuf);
    fftwf_free(sXFftBuf);
    fftwf_free(sHNumBuf);
    free(sHDenBuf);
    fftwf_free(sYBuf);
    
    fftwf_free(sHNumTmpBuf);
    free(sHDenTmpBuf);

    fftwf_free(sfft_input);
    fftwf_free(sfft_output);
    fftwf_free(sfft_plan);
    
    fftwf_free(sifft_input);
    fftwf_free(sifft_output);
    fftwf_free(sifft_plan);
    
    free(eXTmpl);
    free(eXNow);

    free(sDetectFHogCache);
    fftwf_free(sDetectFHogFFTCache);
}

void PVTrack::getPFeatures(sRect r, unsigned char* imgData , int imgH, int imgW,float* featureD)
{
    sMatf  I_Rect;
    sMatc I_Rect_temp, I_Rect_temp_blur;
    sRect extracted_roi;
    int* cnHist = (int*)malloc(sizeof(int)*pPatchSZ[0]*pPatchSZ[1]);
    memset(cnHist,0,sizeof(int)*pPatchSZ[0]*pPatchSZ[1]);

    int cx = r.x + r.width / 2;
    int cy = r.y + r.height / 2;

    extracted_roi.width =  pSzFactorW * pTmplSZ.w;
    extracted_roi.height = pSzFactorH * pTmplSZ.h;

    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    exRoi.x = extracted_roi.x;
    exRoi.y = extracted_roi.y;
    exRoi.width = extracted_roi.width;
    exRoi.height = extracted_roi.height;

    /****************************************************/
    I_Rect_temp = sMatc(extracted_roi.width,extracted_roi.height);
    I_Rect_temp_blur = sMatc(extracted_roi.width,extracted_roi.height);
    I_Rect = sMatf(pTmplSZ.w,pTmplSZ.h);

    int offset = 0;
    for(int i=0; i< pNChannels; i++){
        if(pNChannels == 1) offset = hogIndex; else offset = i;
        unsigned char* srcData = imgData+offset*imgH*imgW;
        float* dstFea = featureD+i*(pPatchSZ[0]*pPatchSZ[1]*pPatchSZ[2]);
        cutRectSimple(srcData, I_Rect_temp.data, imgH, imgW,extracted_roi);
//        avgBlur(I_Rect_temp.data,I_Rect_temp.rows, I_Rect_temp.cols,I_Rect_temp_blur.data,9);
        resize(I_Rect_temp.data, I_Rect_temp.cols, I_Rect_temp.rows,I_Rect.data,pTmplSZ.w,pTmplSZ.h);
        getTrackFhog(I_Rect.data,I_Rect.rows, I_Rect.cols , dstFea, cell_size);

        float*q=dstFea;  //特征添加hanning窗口
        float*p=pHann.data;
        for (int i = 0; i < pPatchSZ[2]*pPatchSZ[0]*pPatchSZ[1]; ++i)
            (*q++)*=  (*p++);
    }

//    save2txtuchar(I_Rect_temp.data,I_Rect_temp.rows,I_Rect_temp.cols,"I_Rect_temp_get_pfeature.txt");
//    save2txt(I_Rect.data,I_Rect.rows,I_Rect.cols,"I_Rect_get_pfeature.txt");

    if(isPHistogram != 0){
        for(int i=0; i< 3; i++){
            unsigned char* srcData = imgData+i*imgH*imgW;
            cutRectSimple(srcData, I_Rect_temp.data, imgH, imgW,extracted_roi);
            resize(I_Rect_temp.data, I_Rect_temp.cols, I_Rect_temp.rows,I_Rect.data,pPatchSZ[0],pPatchSZ[1]);
            for(int k =0; k<pPatchSZ[0]*pPatchSZ[1]; k++){
                cnHist[k] += (((int)(I_Rect.data[k])/8)*pow(32,(2-i)));
            }
        }

        int page = pPatchSZ[0]*pPatchSZ[1];
        float* dstFea = featureD+pNChannels*(pPatchSZ[0]*pPatchSZ[1]*pPatchSZ[2]);
        float*hann =pHann.data;

        for(int k =0; k<page; k++){
            int index = cnHist[k];
            for(int i=0; i<histLen; i++){
                dstFea[k+i*page] = (tableFeature[index*histLen+i])*hann[k];
            }
        }
    }

    free(cnHist);
    I_Rect.release();
    I_Rect_temp.release();
    I_Rect_temp_blur.release();
}

void PVTrack::getSFeatures(sRect rect, unsigned char* imgData , int imgH , int imgW, fftwf_complex* dstFeature){
    float scaleFactor = 0;
    sRect scaleRect;

    sMatc scaleMatTmp;
    sMatf scaleMat;

    unsigned char* scaleMatTmpBuf;
    float* scaleMatBuf;

    unsigned char* scaleMatBig;
    sRect scaleRectBase;

    int scaleFeaWidth = sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen);
    scaleMat = sMatf(sTmplSZ.w,sTmplSZ.h);

    int* cnHist = (int*)malloc(sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);
    memset(cnHist,0,sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);

    ////////////////////////  i = 0 /////////////////////////
    scaleFactor = getFactor(0);
    scaleRect = getScaleRect(rect,scaleFactor);
    scaleRectBase = sRect(scaleRect.x, scaleRect.y,scaleRect.width,scaleRect.height);

    scaleMatTmp = sMatc(scaleRect.width,scaleRect.height);
    scaleMatTmpBuf = (unsigned char*)malloc(sizeof(unsigned char)*scaleRectBase.width*scaleRectBase.height*4);

    int offset = 0;
    for(int ch =0; ch<4; ch++){
        offset = ch;
        unsigned char* srcData = imgData+offset*imgH*imgW;
        unsigned char* dstData = scaleMatTmpBuf+ch*scaleRectBase.width*scaleRectBase.height;
        cutRectSimple(srcData, dstData,imgH,imgW,scaleRectBase);
    }

    scaleMatBig = (unsigned char*)malloc(scaleRectBase.width*scaleRectBase.height*4);
    memcpy(scaleMatBig,scaleMatTmpBuf,scaleRectBase.width*scaleRectBase.height*4);

    for(int i=0; i< sNscales;i++){
        scaleFactor = getFactor(i);
        scaleRect = getScaleRect(rect,scaleFactor);

        scaleMatTmp.cols = scaleRect.width;
        scaleMatTmp.rows = scaleRect.height;

        float* nP = sX.data+i*scaleFeaWidth;
        memset(cnHist,0,sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);
        int offset = 0;
        for(int ch = 0; ch < sNChannels; ch++){
            if(sNChannels == 1) offset=hogIndex; else offset = ch;
            unsigned char* srcData = scaleMatBig + offset*(scaleRectBase.width*scaleRectBase.height);
            cutRectInRect(scaleRectBase , scaleRect, srcData, scaleMatTmp.data);
            resize(scaleMatTmp.data, scaleMatTmp.cols, scaleMatTmp.rows,scaleMat.data,sTmplSZ.w,sTmplSZ.h);

            float* dstData = nP+ch*(sPatchSZ[0]*sPatchSZ[1]*sPatchSZ[2]);
            getScaleTrackFhog(scaleMat.data,scaleMat.rows , scaleMat.cols, dstData,cell_size);
        }

        if(isSHistogram != 0){
            for(int ch = 0; ch < 3; ch++){
                unsigned char* srcData = scaleMatBig + ch*(scaleRectBase.width*scaleRectBase.height);
                cutRectInRect(scaleRectBase , scaleRect, srcData, scaleMatTmp.data);
                resize(scaleMatTmp.data, scaleMatTmp.cols, scaleMatTmp.rows,scaleMat.data,sPatchSZ[0],sPatchSZ[1]);
                for(int k =0; k<sPatchSZ[0]*sPatchSZ[1]; k++){
                    cnHist[k] += (((int)(scaleMat.data[k])/8)*pow(32,(2-ch)));
                }
            }

            int page = sPatchSZ[0]*sPatchSZ[1];
            float* dstFea = nP+sNChannels*(sPatchSZ[0]*sPatchSZ[1]*sPatchSZ[2]);
            for(int k =0; k<page; k++){
                int index = cnHist[k];
                for(int i=0; i<histLen; i++){
                    dstFea[k+i*page] = (tableFeature[index*histLen+i]);
                }
            }
        }

        memcpy(sDetectFHogCache+i*scaleFeaWidth,nP,scaleFeaWidth*sizeof(float));

        float*q=nP;
        for (int j = 0; j <scaleFeaWidth; ++j){
           (*q++) *= (sHann[i]);
        }
    }

    // FFt
    for(int i=0; i < scaleFeaWidth; i++){
        for(int j=0; j<sNscales; j++){
            sfft_input[j][0] = sX.data[j*scaleFeaWidth+i];
            sfft_input[j][1] = 0;
        }

        fftwf_execute(sfft_plan);

        for(int j=0; j<sNscales; j++){
            dstFeature[j*scaleFeaWidth+i][0] = sfft_output[j][0];
            dstFeature[j*scaleFeaWidth+i][1] = sfft_output[j][1];
        }
    }
    memcpy(sDetectFHogFFTCache,dstFeature,sNscales*scaleFeaWidth*sizeof(fftwf_complex));

    free(cnHist);
    free(scaleMatBig);
    free(scaleMatTmpBuf);
    scaleMat.release();
    scaleMatTmp.release();
}


void PVTrack::getSFeaturesInTrain(int factorIndex, sRect rect, unsigned char* imgData , int imgH , int imgW, fftwf_complex* dstFeature){
    float scaleFactor = 0;
    sRect scaleRect;
    sMatc scaleMatTmp;
    sMatf scaleMat;

    int* cnHist = (int*)malloc(sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);
    memset(cnHist,0,sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);

    int scaleFeaWidth = sPatchSZ[0]*sPatchSZ[1]*(sPatchSZ[2]*sNChannels+isSHistogram*histLen);
    scaleMat = sMatf(sTmplSZ.w,sTmplSZ.h);

    scaleFactor = getFactor(0);
    scaleRect = getScaleRect(rect,scaleFactor);

    scaleMatTmp = sMatc(scaleRect.width,scaleRect.height);


    for(int i=0; i< sNscales;i++){
        float* nP = sX.data+i*scaleFeaWidth;

        int cacheHog = getFactorIndex(i) + factorIndex;
        if(cacheHog<=((sNscales-1)/2) && cacheHog>=((1-sNscales)/2)){
            int tmpP = getFactorIndex(cacheHog);
            memcpy(nP,sDetectFHogCache+tmpP*scaleFeaWidth,scaleFeaWidth*sizeof(float));
        }else{
            scaleFactor = getFactor(i);
            scaleRect = getScaleRect(rect,scaleFactor);
            scaleMatTmp.cols = scaleRect.width;
            scaleMatTmp.rows = scaleRect.height;
            memset(cnHist,0,sizeof(int)*sPatchSZ[0]*sPatchSZ[1]);
            int offset = 0;
            for(int ch=0; ch<sNChannels;ch++){
                if(sNChannels == 1) offset=hogIndex; else offset = ch;
                unsigned char* srcData = imgData+offset*imgH*imgW;
                float* dstData = nP+ch*(sPatchSZ[0]*sPatchSZ[1]*sPatchSZ[2]);
                cutRectSimple(srcData, scaleMatTmp.data,imgH,imgW,scaleRect);
                resize(scaleMatTmp.data, scaleMatTmp.cols, scaleMatTmp.rows,scaleMat.data,sTmplSZ.w,sTmplSZ.h);
                getScaleTrackFhog(scaleMat.data,scaleMat.rows , scaleMat.cols, dstData,cell_size);
            }

            if(isSHistogram != 0){
                for(int ch=0; ch<3;ch++){
                    unsigned char* srcData = imgData+ch*imgH*imgW;
                    cutRectSimple(srcData, scaleMatTmp.data,imgH,imgW,scaleRect);
                    resize(scaleMatTmp.data, scaleMatTmp.cols, scaleMatTmp.rows,scaleMat.data,sPatchSZ[0],sPatchSZ[1]);
                    for(int k =0; k<sPatchSZ[0]*sPatchSZ[1]; k++){
                        cnHist[k] += (((int)(scaleMat.data[k])/8)*pow(32,(2-ch)));
                    }
                }

                int page = sPatchSZ[0]*sPatchSZ[1];
                float* dstFea = nP+sNChannels*(sPatchSZ[0]*sPatchSZ[1]*sPatchSZ[2]);
                for(int k =0; k<page; k++){
                    int index = cnHist[k];
                    for(int i=0; i<histLen; i++){
                        dstFea[k+i*page] = (tableFeature[index*histLen+i]);
                    }
                }
            }
        }

        float*q=nP;
        for (int j = 0; j <scaleFeaWidth; ++j){
           (*q++) *= (sHann[i]);
        }
    }

    // FFt
    for(int i=0; i < scaleFeaWidth; i++){
        for(int j=0; j<sNscales; j++){
            sfft_input[j][0] = sX.data[j*scaleFeaWidth+i];
            sfft_input[j][1] = 0;
        }

        fftwf_execute(sfft_plan);

        for(int j=0; j<sNscales; j++){
            dstFeature[j*scaleFeaWidth+i][0] = sfft_output[j][0];
            dstFeature[j*scaleFeaWidth+i][1] = sfft_output[j][1];
        }
    }
    memcpy(sDetectFHogFFTCache,dstFeature,sNscales*scaleFeaWidth*sizeof(fftwf_complex));

    free(cnHist);
    scaleMat.release();
    scaleMatTmp.release();
}


void PVTrack::getEstimateFeatures(sRect r, unsigned char* imgData , int imgH, int imgW,float* featureD)
{
    sMatf  I_Rect;
    sMatc I_Rect_temp;
    sRect tmpRoi(r.x,r.y,r.width,r.height);

    /****************************************************/
    I_Rect_temp = sMatc(tmpRoi.width,tmpRoi.height);
    I_Rect = sMatf(sTmplSZ.w,sTmplSZ.h);

    int offset = 0;
    for(int i=0; i< 3; i++){
        offset = i;
        unsigned char* srcData = imgData+offset*imgH*imgW;
        float* dstFea = featureD+i*(sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]);
        cutRectSimple(srcData, I_Rect_temp.data, imgH, imgW,tmpRoi);
        resize(I_Rect_temp.data, I_Rect_temp.cols, I_Rect_temp.rows,I_Rect.data,sTmplSZ.w,sTmplSZ.h);
        getTrackFhog(I_Rect.data,I_Rect.rows, I_Rect.cols , dstFea, cell_size);
    }

    I_Rect.release();
    I_Rect_temp.release();
}

void PVTrack::getEstimateScore()
{
    float* a = eXTmpl;
    float* b = eXNow;
    int LEN = sPatchSZ[0]*sPatchSZ[1]*pPatchSZ[2]*3;
    float score = 0;
    float tmp = 0.0;
    for(int i=0; i< LEN; i++){
        tmp = (*a++)-(*b++);
        score += (tmp*tmp);
    }

    score = 100-(score*1000/(float)LEN);
    score = (score-60)*2.5;

    reliable = (int)score;
    // printf("score:   %.3f\n",score);


//    if(reliable>75){
//        for(int i = 0;i<LEN ;++i){
//            eXTmpl[i] = (1 - pYita/3) * eXTmpl[i] + (pYita/3) * eXNow[i];
//        }
//    }

}

void PVTrack::gaussianCorrelation(float* f1data , float* f2data, float* resData, int width , int height, int depth, float sigma){
    int ss = height* width ;
    float* c = new float[ss];
    memset(c,0,sizeof(float)*ss);

    int ind = 0;
    float* x1=f1data;
    float* x2=f2data;

    float *  x1aux;
    float *  x2aux;

    for (int i = 0; i < (depth*pNChannels+isPHistogram*histLen) ; ++i){

        x1aux = x1+i*width*height;
        x2aux = x2+i*width*height;

        for (int i = 0; i < height; ++i){
            for (int j = 0; j < width; ++j){
                ind = i *  width + j;
                pfft_input[ind][0] = x1aux[ind];
                pfft_input[ind][1] = 0;
            }
        }
        fftwf_execute(pfft2_plan);

        for (int i = 0; i < height; ++i){
            for (int j = 0; j < width; ++j){
                ind = i *  width + j;
                ptmp[ind][0]= pfft_output[ind][0];
                ptmp[ind][1]= pfft_output[ind][1];
                pfft_input[ind][0] = x2aux[ind];
                pfft_input[ind][1] = 0;
            }
        }
        fftwf_execute(pfft2_plan);

        mulSpectrums(ptmp,pfft_output,pifft_input, width, height ,1);

        fftwf_execute(pifft2_plan);
        float* rc = new float[ss];

        float* ptr = rc;
        for (int i = 0; i < ss; ++i){
            pifft_output[i][0]*= 1.f /ss;
            (*ptr++)= pifft_output[i][0];
        }
        rearrange(rc,width, height);
//        rearrangeAll(rc,width, height);

        for(int i =0;i< ss;++i){
            c[i]+=rc[i];
        }
        delete (rc);
    }

    int s =(width*height*(depth*pNChannels+isPHistogram*histLen));

    memset(resData,0,sizeof(float)*ss);
    float sumx1=0;
    float sumx2=0;
    for(int i = 0; i< s ;++i){
        sumx1 += (*x1)*(*x1);x1++;
        sumx2 += (*x2)*(*x2);x2++;
    }
    float SUM = sumx2+sumx1;

    float * ptr = c;
    float * qtr = resData;


    for(int i = 0; i< ss ;++i){
        (*qtr ) = max((SUM- 2*(*ptr++))/s, float(1e-19));
        (*qtr ) = exp((- (*qtr ) / (sigma * sigma)));
        qtr++;
    }

    delete c;
}

