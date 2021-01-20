#include "imgProcess.h"
#include "UltraLightFastGenericGaceDetector1MB.h"
#include "opencv2/opencv.hpp"
#include "handDetect.h"
#include "handRecognition.h"
#include "shoulderDetect.h"
#include "align.h"


#include<stdio.h>
#include<string>
#include<iostream>
#include<dirent.h>
#include <vector>
#include"objectTrack.h"
#include "math.h"

#include "tool_mn.h"

using namespace std;
using namespace cv;

int angleCamera = 0;
int orientationCamera = 0;

kalman1_state state1;
kalman1_state state2;
kalman1_state state3;
kalman1_state state4;

kalman1_state state11;
kalman1_state state22;
kalman1_state state33;
kalman1_state state44;

void kalman1_init(kalman1_state *state, float init_x, float init_p)
{
    state->x = init_x;
    state->p = init_p;
    state->A = 1;
    state->H = 1;
    state->q = 20;//10e-6;  /* predict noise convariance */
    state->r = 20;//10e-5;  /* measure error convariance */
}

float kalman1_filter1(kalman1_state *state, float track, float z_measure)
{
    /* Predict */
    //state->x = state->A * state->x;
    state->q = 1;  // 10
    state->r = 5;  // 10
    state->x = state->A * track;
    state->p = state->A * state->A * state->p + state->q;  /* p(n|n-1)=A^2*p(n-1|n-1)+q */

    /* Measurement */
    state->gain = state->p * state->H / (state->p * state->H * state->H + state->r+0.1);
    state->x = state->x + state->gain * (z_measure - state->H * state->x);
    state->p = (1 - state->gain * state->H) * state->p;

    return state->x;
}

float kalman1_filter2(kalman1_state *state, float track, float z_measure, int diff)
{
    state->q = 1;
    //state->r = 1000000;
    if(diff>10)
    {
        state->r = 1;
    }
    else if(diff>5)
    {
        state->r = 1;
    }
    else if(diff>3)
    {
        state->r = 2;
    }
    else
    {
        state->r = 10;
    }

    /* Predict */
    //state->x = state->A * state->x;
    state->x = state->A * track;
    state->p = state->A * state->A * state->p + state->q;  /* p(n|n-1)=A^2*p(n-1|n-1)+q */

    /* Measurement */
    state->gain = state->p * state->H / (state->p * state->H * state->H + state->r+0.1);
    state->x = state->x + state->gain * (z_measure - state->H * state->x);
    state->p = (1 - state->gain * state->H) * state->p;

    return state->x;
}

//float kalman1_filter2(kalman1_state *state, float z_measure, int diff)
//{
//    state->q = 1;
//    if(diff>15)
//    {
//        state->r = 1;
//    }
//    else if(diff>10)
//    {
//        state->r = 2;
//    }
//    else if(diff>5)
//    {
//        state->r = 9;
//    }
//    else
//    {
//        state->r = 50;
//    }
//
//    /* Predict */
//    state->x = state->A * state->x;
//    //state->x = state->A * track;
//    state->p = state->A * state->A * state->p + state->q;  /* p(n|n-1)=A^2*p(n-1|n-1)+q */
//
//    /* Measurement */
//    state->gain = state->p * state->H / (state->p * state->H * state->H + state->r+0.1);
//    state->x = state->x + state->gain * (z_measure - state->H * state->x);
//    state->p = (1 - state->gain * state->H) * state->p;
//
//    return state->x;
//}


float iouFaceTrack(cv::Rect box0, cv::Rect box1)
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;
    
    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    
    if (u <= 0.0) return 0.0f;
    else          return i/u;
}


fusionInf fusionIou(std::vector<cv::Rect> faces, float* trackResult, int w, int h)
{
    fusionInf faceInformation;
    /* 依据检测结果进行跟踪重叠判断 */
    int faceid = -1;  // 检测和跟踪融合的有效人脸id
    int ii = 0;       // 人脸的id
    float faceIouMax = 0.00001;  // 检测和跟踪的融合参数

    for(cv::Rect face: faces)
    {
        //printf("have face11\n");
        cv::Rect tracktmp;
        tracktmp.x = (int)trackResult[0];
        tracktmp.y = (int)trackResult[1];
        tracktmp.width = (int)trackResult[2];
        tracktmp.height = (int)trackResult[3];
        
        cv:Rect facetmp;
        facetmp.x = (int)((float)face.x/320*w);
        facetmp.y = (int)((float)face.y/240*h);
        facetmp.width = (int)((float)face.width/320*w);
        facetmp.height = (int)((float)face.height/240*h);
        
        printf("[mnn2] facetmp is (%d, %d, %d, %d)\n",facetmp.x, facetmp.y, facetmp.width, facetmp.height );
        printf("[mnn2] tracktmp is (%d, %d, %d, %d)\n",tracktmp.x, tracktmp.y, tracktmp.width, tracktmp.height );
 
        float faceIou = iouFaceTrack(facetmp, tracktmp);
        if( faceIou>faceIouMax )
        {
            faceIouMax = faceIou;
            faceid = ii;
        }
        ii++;
    }
    
    faceInformation.faceid = faceid;
    faceInformation.iou = faceIouMax;
    
    return faceInformation;
}


/* 人脸检测与landmark */
int detFace(cv::Mat input_rgb, const char* path, std::vector<cv::Rect> &faces,
std::vector<landmarkFace> &landmarkBoxResult, int detectState)
{
    /* load model */
    static UltraLightFastGenericGaceDetector1MB ultralightfastgenericgacedetector1MB;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        ultralightfastgenericgacedetector1MB.load(model_path);
        is_model_prepared = true;
    }
    
    /* detect face */
    ultralightfastgenericgacedetector1MB.detect(input_rgb, faces, landmarkBoxResult, detectState);
    return 0;
}


/* 手势检测(人脸局部区域) */
int detectHand(cv::Mat input_rgb, const char* path, std::vector<cv::Rect> &faces)
{
    static HandDetect HandDetect;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        HandDetect.load(model_path);
        is_model_prepared = true;
    }
    HandDetect.detect(input_rgb, faces);

    return 0;
}

/* 手势识别*/
int recognitionHand(cv::Mat input_rgb, const char* path)
{
    static HandRecognition HandRecognition;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        HandRecognition.load(model_path);
        is_model_prepared = true;
    }
    int handFlag = HandRecognition.recognition(input_rgb);

    return handFlag;
}

/* 头肩检测 */
int shoulderPerson(cv::Mat input_rgb, const char* path, std::vector<cv::Rect> &shoulders, int flag)
{
    static ShoulderDetect ShoulderDetect;
    static bool is_model_prepared = false;
    if (false == is_model_prepared)
    {
        const char* model_path = path;
        ShoulderDetect.load(model_path);
        is_model_prepared = true;
    }
    ShoulderDetect.detect(path, input_rgb, shoulders, flag);

    return 0;
}


/* 存储图像 */
int saveImg(cv::Mat input_rgb, const std::string &path, int i )
{

        std::string result_name = path+"/img_result" + std::to_string(11) + ".jpg";
        //cv::imwrite(result_name, input_rgb);

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);  //选择jpeg
        compression_params.push_back(100); //在这个填入你要的图片质量

    
        cv::imwrite(result_name, input_rgb, compression_params);
//        LOGD("write image ok");
    return 0;
}

/* 读取图像 */
 cv::Mat loadImg(const std::string &path, cv::Mat img)
{
    std::string imgpath = path + "/1.jpg";
    printf("path is %s\n", imgpath.c_str());

    img = cv::imread(imgpath);
    printf("img w h is %d, %d\n", img.cols, img.rows);
    return img;
}

/* 存储图像 */
void saveimg(const std::string &path, cv::Mat img)
{
//    std::string imgpath = "/Users/scott.mao/Desktop/result5.jpg";
//    printf("path is %s\n", imgpath.c_str());

    std::string result_name = path+"/img_result" + std::to_string(95) + ".jpg";
    cv::imwrite(result_name, img);
}


/* 读取图像路径 */
//int imglist(const std::string &path, char* imgname, int frameid )
int imglist(const std::string &path, const std::string &path1,char* imgname, int frameid )
{
//    sprintf(imgname, "/Users/powervision/Desktop/img/%.4d.png", frameid);
    std::string  imgname1 = path + path1;
    sprintf(imgname, imgname1.c_str(), frameid);
    //printf("1\n");
    
    return 0;
}

/* 图像转换 */
void cvMat32sMat(cv::Mat img, sMatc retImager1, sMatc retImageg1, sMatc retImageb1, unsigned char* rgb)
{
    cv::Mat newImage1;
    cv::Mat grayImgae1;
    cvtColor(img, grayImgae1, COLOR_BGR2GRAY);
    
    vector<Mat> channels;
    split(img, channels);
    cv::Mat b = channels.at(0);
    cv::Mat g = channels.at(1);
    cv::Mat r = channels.at(2);
    
    int w = b.cols;
    int h = b.rows;
    
    retImager1.cols = w;
    retImager1.rows = h;
    retImageg1.cols = w;
    retImageg1.rows = h;
    retImageb1.cols = w;
    retImageb1.rows = h;
    
    memcpy(retImager1.data, r.data, sizeof(unsigned char)*w*h);
    memcpy(retImageg1.data, g.data, sizeof(unsigned char)*w*h);
    memcpy(retImageb1.data, b.data, sizeof(unsigned char)*w*h);
    
    memcpy(rgb,         b.data, sizeof(unsigned char)*w*h);
    memcpy(rgb+(w*h),   g.data, sizeof(unsigned char)*w*h);
    memcpy(rgb+(w*h*2), r.data, sizeof(unsigned char)*w*h);
    memcpy(rgb+(w*h*3), grayImgae1.data, sizeof(unsigned char)*w*h);
    
}


/* 接收控制状态  */
int xmin;
int ymin;
int width;
int height;
cmd_status command_status;          // 当前帧状态信息
cmd_status command_hist_status;     // 历史帧状态信息
int angleUpdateFlag_current=0;        // 当前帧更新角度的标志位，更新角度此状态位置1
int acceptData(std::vector<accept_data> &initbox, int h, int w)
{
    /* 获取当前帧的跟踪角度信息，暂未为历史信息, 角度信息与方向信息 */
    command_hist_status.angleCamera = command_status.angleCamera;
    command_hist_status.orientationCamera = command_status.orientationCamera;
    
    /* 接收app初始跟踪坐标 */
    xmin = initbox[0].x*w;
    ymin = initbox[0].y*h;
    width = initbox[0].w*w;
    height = initbox[0].h*h;
    printf("accept init location is %d, %d, %d, %d\n",xmin,ymin,width,height);
    
    /* 保护处理，app发送目标过小或者负数则不进行跟随 */
    if( width<10 || height<10 ) return 0;
    
    /* 将0-1坐标转换为像素坐标，竖屏左上角点坐标， 后置存在镜像关系 */
    if( (command_status.angleCamera==0 && command_status.orientationCamera==1) ||
        (command_status.angleCamera==180 && command_status.orientationCamera==1) )  // front right/left
    {
        xmin = initbox[0].x*w;
        ymin = initbox[0].y*h;
        width = initbox[0].w*w;
        height = initbox[0].h*h;
    }
    else if( (command_status.angleCamera==-90 && command_status.orientationCamera==1) ||
             (command_status.angleCamera==90 && command_status.orientationCamera==1) )
    {
        xmin = initbox[0].x*h;
        ymin = initbox[0].y*w;
        width = initbox[0].w*h;
        height = initbox[0].h*w;
    }
    
    if( (command_status.angleCamera==0 && command_status.orientationCamera==0) ||
        (command_status.angleCamera==180 && command_status.orientationCamera==0) )  // rear right/left
    {
        xmin = w - initbox[0].x*w - initbox[0].w*w;
        ymin = initbox[0].y*h;
        width = initbox[0].w*w;
        height = initbox[0].h*h;
    }
    else if( (command_status.angleCamera==-90 && command_status.orientationCamera==0) ||
             (command_status.angleCamera==90 && command_status.orientationCamera==0) )
    {
        xmin = h - initbox[0].x*h - initbox[0].w*h;
        ymin = initbox[0].y*w;
        width = initbox[0].w*h;
        height = initbox[0].h*w;
    }
  
    

    printf("xmin, ymin, width, height is %d, %d, %d, %d\n",xmin, ymin, width, height);
    
    /* 开始跟随和手势检测的状态开关 */
    command_status.track_status = initbox[0].track_status;
    //command_status.hand_status = initbox[0].hand_status;
    command_status.initflag = initbox[0].initflag;
    
    printf("[mnn3] command_status.track_status is (%d, %d)\n",command_status.track_status, command_status.hand_status );
    
    /* 初始跟随，切换角度状态位置0 */
    angleUpdateFlag_current = 0;
    
    return 1;
}

/* 控制手势的开启和关闭以及跟随的关闭 */
int acceptControlParam(std::vector<accept_data> &initbox)
{
    command_status.track_status = initbox[0].track_status;
    command_status.hand_status = initbox[0].hand_status;
    command_status.initflag = initbox[0].initflag;
    printf("accept track_status: %d, hand_track_state:%d, initflag:%d\n", command_status.track_status, command_status.hand_status, command_status.initflag);
    
    printf("[mnn3] command_status.track_status is (%d, %d)\n",command_status.track_status, command_status.hand_status );
    
    return 1;
}

void getCameraParam(int angle1, int orientation1)
{
    /* 获取相机最新的旋转角度信息 */
    command_status.angleCamera = angle1;
    command_status.orientationCamera = orientation1;
    
    /* 切换角度，重置标志位 */
    angleUpdateFlag_current = 1;
    
    /* 显示当前切换的角度与姿态信息 */
    printf("angleCamera orientCamera is %d, %d\n", command_status.angleCamera, command_status.orientationCamera);
}

/* 左上角点转当前屏幕 */
void rotionTrackLocation(int* result, int w, int h, int angle, int orient)
{
    float tmp[4];
    tmp[0] = result[0];
    tmp[1] = result[1];
    tmp[2] = result[2];
    tmp[3] = result[3];
    int w1, h1;
    
    /* 当前屏幕宽高 */
    if( angle==0 || angle==180 )
    {
        w1 = h;
        h1 = w;
    }
    else if( angle==-90 || angle==90)
    {
        w1 = w;
        h1 = h;
    }
    
    w1 = 640;
    h1 = 640;
    
    /* fornt */
    if( command_status.orientationCamera==1 )
    {
        if( command_status.angleCamera==0 )  // 0 right
        {
            result[0] = h1 - tmp[1] - tmp[3];
            result[1] = tmp[0];
            result[2] = tmp[3];
            result[3] = tmp[2];
            
        }
        else if( command_status.angleCamera==-90 ) // up
        {
            //sudo
        }
        else if( command_status.angleCamera==180 ) // left
        {
            result[0] = tmp[1];
            result[1] = w1 - tmp[0] - tmp[2];
            result[2] = tmp[3];
            result[3] = tmp[2];
        }
        else if( command_status.angleCamera==90 ) // down
        {
            result[0] = w - tmp[0] - tmp[2];
            result[1] = h - tmp[1] - tmp[3];
            result[2] = result[2];
            result[3] = result[3];
        }
    }
}


PVTrack pvTrack;
float result1[6] = {0,0,0,0,0,0};
int detectId = 0;  // 图像帧数或者是跟踪帧数
int fusionDetect_delay = 5;
int initflag_update = 1;
int handflag = 1;
int handdelay = 0;   // 跟踪手势识别的时间间隔
int angleUpdateFlag = 0;
float resultApp[4];
float resultApp_hist[4];


struct timespec time1={0,0};
struct timespec time2={0,0};

/* 跟踪状态判断 */
pvTrackState trackState;
int rotateTrackFlag;
int pv_visual_main(unsigned char* pixels, int h, int w, const char* path, std::vector<MNNRect> &faceResult, std::vector<MNNRect> &result)
{
    if( angleUpdateFlag_current==1 )
    {
        rotateTrackFlag = 1;
    }else{
        rotateTrackFlag = 0;
    }
    
    //return 0;
    clock_gettime(CLOCK_REALTIME, &time1);
    
    int img_h = h;
    int img_w = w;
    /* 当前先默认开始手势识别开关 */
    //command_status.hand_status=0;
    //command_status.track_status=1;
    
    /* 跟踪手势识别的间隔与默认的手势识别结果状态 */
    handdelay++;
    handflag = 1;
    
    /* 视频流测试接口 */
    cv::Mat frame(h, w, CV_8UC4, pixels);
    cv::flip(frame, frame, 1);
    cv::Mat input_rgb, input_rgb1, input_rgb2, input_rgb3;
    cv::cvtColor(frame, input_rgb1, cv::COLOR_BGRA2RGB);
    
    //return 0;
    
    
    /* 判断当前手机是否旋转，1表示旋转,旋转完成后置2 */
    if( angleUpdateFlag_current != angleUpdateFlag && rotateTrackFlag == 1 )
    {
        angleUpdateFlag = angleUpdateFlag_current;
        angleUpdateFlag_current = 2;
    }
    //printf("angleUpdateFlag is %d\n", angleUpdateFlag);
    
    /*获取当前最新的角度信息，这里hist消息是在发送app坐标之后才进行更新 */
    if( angleUpdateFlag == 1  && rotateTrackFlag == 1)
    {
        rotationImg(input_rgb1, input_rgb2, command_status.angleCamera, command_status.orientationCamera);
        imgPpading(input_rgb2, input_rgb3, command_status.angleCamera, command_status.orientationCamera);
    }else{
        rotationImg(input_rgb1, input_rgb2, command_hist_status.angleCamera, command_hist_status.orientationCamera);
        imgPpading(input_rgb2, input_rgb3, command_hist_status.angleCamera, command_hist_status.orientationCamera);
    }
    
    /* 保证输入图像是vga */
    int imgW = 640;
    int imgH = 480;
    cv::resize(input_rgb3, input_rgb, cv::Size(imgW,imgH));
       
    sMatc framer;
    sMatc frameg;
    sMatc frameb;
    unsigned char* framerbuf = (unsigned char*)malloc(imgW*imgH);
    unsigned char* framegbuf = (unsigned char*)malloc(imgW*imgH);;
    unsigned char* framebbuf = (unsigned char*)malloc(imgW*imgH);;
    unsigned char* framergb = (unsigned char*)malloc(imgW*imgH*4);
    framer = sMatc(imgW,imgH,framerbuf);
    frameg = sMatc(imgW,imgH,framegbuf);
    frameb = sMatc(imgW,imgH,framebbuf);
    
    /* 依据手势或者跟随判断是否进行人脸检测或者头肩检测 */
    int handflag = 1;  // 当前手势识别的状态，默认为无效状态
    std::vector<cv::Rect> faces;
    std::vector<cv::Rect> shoulders;
    std::vector<landmarkFace> landmarkBoxResult;
    
    
    clock_gettime(CLOCK_REALTIME, &time2);
    printf("[MNN] time 1 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
    
    printf("[MNN] detectId is %d\n", detectId);
    
    /* face track 1 */
    //command_status.orientationCamera=1;
    if( command_status.orientationCamera==1 )
    {
        /* 全图手势检测或者跟随手势检测 */
        if( (command_status.track_status!=1)) // 进入全图手势检
        {
            if( detectId%10==0 || detectId<200 )   // 抽帧进行计算
            {
                /* 人脸检测 */
                //clock_gettime(CLOCK_REALTIME,&time1);
                detFace(input_rgb, path, faces, landmarkBoxResult, 1);
                //clock_gettime(CLOCK_REALTIME,&time2);
                //printf("[MNN1] faces.size is %d\n", faces.size());
                //printf("[MNN1] global face TIME : %d ms!\n",(time2.tv_sec-time1.tv_sec)*1000+(time2.tv_nsec-time1.tv_nsec)/1000000);
            
                /* 只有检测则进行全图手势检测 */
                if( command_status.hand_status==1 )
                {
                    for(cv::Rect face: faces)
                    {
                        //printf("1\n");
                        /* 获取手势检测区域 */
                        cv::Rect handbox;
                        float face_imgw = 320.0;
                        float face_imgh = 240.0;
                        handbox.x = (int)(face.x/face_imgw*input_rgb.cols);
                        handbox.y = (int)(face.y/face_imgh*input_rgb.rows);
                        handbox.width = (int)(face.width/face_imgw*input_rgb.cols);
                        handbox.height = (int)(face.height/face_imgh*input_rgb.rows);
                        handbox.x = handbox.x - handbox.width*2;
                        handbox.y = handbox.y - handbox.height*0.5;
                        handbox.width = handbox.width*5;
                        handbox.height = handbox.height*5;
                        
                        if( handbox.x<1 ) handbox.x = 2;
                        if( handbox.y<1 ) handbox.y = 2;
                        if( (handbox.x+handbox.width)>input_rgb.cols-2 ) handbox.width = input_rgb.cols-handbox.x;
                        if( (handbox.y+handbox.height)>input_rgb.rows-2 ) handbox.height = input_rgb.rows-handbox.y;
                        //printf("handbox11 is (%d, %d, %d, %d)\n", handbox.x, handbox.y, handbox.width, handbox.height);
                        if( handbox.width<0 || handbox.height<0 ) continue;
                        
                        /* 图像填充，保证检测图像的畸变问题 */
                        int top, bottom, left, right;
                        cv::Mat handimg1 = input_rgb(handbox);
                        if( handimg1.cols>handimg1.rows )
                        {
                            top = 0;
                            bottom = handimg1.cols-handimg1.rows;
                            left = 0;
                            right =0;
                        }else
                        {
                            top = 0;
                            bottom = 0;
                            left = 0;
                            right = handimg1.rows - handimg1.cols;
                        }
                        cv::Mat handing;
                        copyMakeBorder(handimg1, handing, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(0, 0, 0));
                        
                        /* 手势检测 */
                        int hand_imgw = handing.cols;
                        int hand_imgh = handing.rows;
                        std::vector<cv::Rect> hands;
                        detectHand(handing, path, hands);
                        
                        /* 抠图进行手势识别 */
                        for(cv::Rect hand: hands)
                        {
                            hand.x = fmax((float)hand.x/176*hand_imgw+handbox.x, 2);
                            hand.y = fmax((float)hand.y/176*hand_imgh+handbox.y, 2);
                            hand.width = fmin((float)hand.width/176*hand_imgw, hand_imgw);
                            hand.height = fmin((float)hand.height/176*hand_imgh, hand_imgh);
                            cv::Rect vis_hand;
                            vis_hand.x = fmax(hand.x-hand.width*0.1, 2);
                            vis_hand.y = fmax(hand.y-hand.height*0.1, 2);
                            vis_hand.width = int(fmin(hand.x+hand.width*1.1, input_rgb.cols) - hand.x);
                            vis_hand.height = int(fmin(hand.y+hand.height*1.1, input_rgb.rows) - hand.y);
                            if( vis_hand.width<0 || vis_hand.height<0 ) continue;
                            
                            cv::Mat handimg_r = input_rgb(vis_hand);
                            
                            /* 手势识别 */
                            /* 0 video; 2 picture; 3 group; 4 track; 1 nohand */
                            handflag = recognitionHand(handimg_r, path);
                            //printf("handflag is %d\n", handflag);
                            /* 开启跟随 */
                            if( handflag==3 )
                            {
                               /* 获取人脸坐标 */
                                float face_imgw = 320.0;
                                float face_imgh = 240.0;
                                cv::Rect trackbox;
                                trackbox.x = (int)(face.x/face_imgw*input_rgb3.cols);
                                trackbox.y = (int)(face.y/face_imgh*input_rgb3.rows);
                                trackbox.width = (int)(face.width/face_imgw*input_rgb3.cols);
                                trackbox.height = (int)(face.height/face_imgh*input_rgb3.rows);
                                /* 跟踪参数设置 */
                                xmin = trackbox.x;
                                ymin = trackbox.y;
                                width = trackbox.width;
                                height = trackbox.height;
                                if( command_status.track_status==0 && handdelay>80)  // 开启跟随
                                {
                                    handdelay = 0;
                                    command_status.track_status = 1;
                                    command_status.initflag = 1;
                                }else if( handdelay>80 ){
                                    handdelay = 0;
                                    command_status.track_status = 0;   // 关闭跟随
                                    command_status.initflag = 0;
                                }
                                /* 手势控制说明，当前目标是人 */
                                trackState.objectLabel = 1;
                            }
                            if( handflag!= 1 ) break;
                        }
                        if( handflag!= 1 ) break;
                    }
                }
            }
        }
        else if( (command_status.track_status==1) ) // 进入跟踪手势检测
        {
            if( detectId%10==0)   // 抽帧进行计算
            {
                /* 获取人脸检测区域,6倍区域 */
                cv::Rect regionBox;
                regionBox.x = result1[0] - result1[2]*2.5;
                regionBox.y = result1[1] - result1[3]*2.5;
                regionBox.width = result1[2]*6;
                regionBox.height = result1[3]*6;
                if( regionBox.x<1 ) regionBox.x = 2;
                if( regionBox.y<1 ) regionBox.y = 2;
                if( (regionBox.x+regionBox.width)>input_rgb.cols-2 ) regionBox.width = input_rgb.cols-regionBox.x;
                if( (regionBox.y+regionBox.height)>input_rgb.rows-2 ) regionBox.height = input_rgb.rows-regionBox.y;
                if( regionBox.width<0 || regionBox.height<0 )
                {
                    regionBox.x = 1;
                    regionBox.y = 1;
                    regionBox.width = 20;
                    regionBox.height = 20;
                }
                //printf("[MNN1] regionBox is (%d, %d, %d, %d)\n", regionBox.x, regionBox.y, regionBox.width, regionBox.height);
                cv::Mat regionBox_img = input_rgb(regionBox);
                clock_gettime(CLOCK_REALTIME,&time1);
                /* 局部人脸检测 */
                std::vector<cv::Rect> faces_tmp;
                //detFace(input_rgb, path, faces, landmarkBoxResult,2);
                detFace(regionBox_img, path, faces_tmp, landmarkBoxResult,2);
                clock_gettime(CLOCK_REALTIME,&time2);
                //printf("[MNN1] faces.size is %d\n", faces_tmp.size());
                printf("[MNN1] local face TIME : %d ms!\n",(time2.tv_sec-time1.tv_sec)*1000+(time2.tv_nsec-time1.tv_nsec)/1000000);


                /* 局部坐标转全局坐标 */
                float regionBox_imgW = 160.0;
                float regionBox_imgH = 120.0;
                for(cv::Rect face: faces_tmp)
                {
                    /* vga坐标 */
                    face.x = (float)face.x / regionBox_imgW * regionBox_img.cols + regionBox.x;
                    face.y = (float)face.y / regionBox_imgH * regionBox_img.rows + regionBox.y;
                    face.width = (float)face.width / regionBox_imgW * regionBox_img.cols;
                    face.height = (float)face.height / regionBox_imgW * regionBox_img.cols;
                    
                    /* 320坐标 */
                    face.x = face.x * 0.5;
                    face.y = face.y * 0.5;
                    face.width = face.width * 0.5;
                    face.height = face.height * 0.5;
                    faces.push_back(face);
                    
                }
                
                /* 跟踪区域的手势检测 */
                if( command_status.hand_status==1 )
                {
                    /* 获取手势检测区域 */
                    cv::Rect handbox;
                    handbox.x = result1[0] - result1[2]*2;
                    handbox.y = result1[1] - result1[3]*0.5;
                    handbox.width = result1[2]*5;
                    handbox.height = result1[3]*4;
                    
                    if( handbox.x<1 ) handbox.x = 2;
                    if( handbox.y<1 ) handbox.y = 2;
                    if( (handbox.x+handbox.width)>input_rgb.cols-2 ) handbox.width = input_rgb.cols-handbox.x;
                    if( (handbox.y+handbox.height)>input_rgb.rows-2 ) handbox.height = input_rgb.rows-handbox.y;
                    if( handbox.width<0 || handbox.height<0 )
                    {
                        handbox.x = 1;
                        handbox.y = 1;
                        handbox.width = 20;
                        handbox.height = 20;
                    }
           
                    /* 图像填充，保证检测图像的畸变问题 */
                    int top, bottom, left, right;
                    cv::Mat handimg1 = input_rgb(handbox);
                    if( handimg1.cols>handimg1.rows )
                    {
                        top = 0;
                        bottom = handimg1.cols-handimg1.rows;
                        left = 0;
                        right =0;
                    }else
                    {
                        top = 0;
                        bottom = 0;
                        left = 0;
                        right = handimg1.rows - handimg1.cols;
                    }
                    cv::Mat handing = handimg1;
                    //cv::Mat handing;
                    //copyMakeBorder(handimg1, handing, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(0, 0, 0));
                        
                    /* 手势检测 */
                    int handw = handing.cols;
                    int handh = handing.rows;
                    std::vector<cv::Rect> hands;
                    detectHand(handing, path, hands);
             
                    /* 抠图进行手势识别 */
                    for(cv::Rect hand: hands)
                    {
                        hand.x = fmax((float)hand.x/176*handw+handbox.x, 2);
                        hand.y = fmax((float)hand.y/176*handh+handbox.y, 2);
                        hand.width = fmin((float)hand.width/176*handw, handw);
                        hand.height = fmin((float)hand.height/176*handh, handh);
                        
                        // 手势太靠边不识别
                        //if( handing.cols < (hand.x+hand.width) ) continue;
                        //if( hand.x<2 ) continue;
                        //if( handing.rows < (hand.y+hand.height) ) continue;
                        //if( hand.y<2 ) continue;
                        
                        cv::Rect vis_hand;
                        vis_hand.x = fmax(hand.x-hand.width*0.1, 2);
                        vis_hand.y = fmax(hand.y-hand.height*0.1, 2);
                        vis_hand.width = int(fmin(hand.x+hand.width*1.1, input_rgb.cols) - hand.x);
                        vis_hand.height = int(fmin(hand.y+hand.height*1.1, input_rgb.rows) - hand.y);
                        if( vis_hand.width<0 || vis_hand.height<0 ) continue;
                        
                        /* 手势识别 */
                        cv::Mat handimg_r = input_rgb(vis_hand);
                        handflag = recognitionHand(handimg_r, path);
                        //printf("handflag is %d\n", handflag);
                        /* 开启跟随 */
                        if( handflag==3 )
                        {
                            if( handdelay>100 ){  // 关闭跟随
                                handdelay = 0;
                                command_status.track_status = 0;
                                command_status.initflag = 0;
                            }
                        }
                        if( handflag!= 1 ) break;
                    }
                }
            }
        }
        
        /* 发送人脸检测结果 */
        for( cv::Rect face: faces )
        {
            float imgw_face = 320.0;
            float imgh_face = 240.0;
            MNNRect facetmp;
            facetmp.x = (int) (face.x / imgw_face * input_rgb3.cols);
            facetmp.y = (int) (face.y / imgh_face * input_rgb3.rows);
            facetmp.w = (int) (face.width / imgw_face * input_rgb3.cols);
            facetmp.h = (int) (face.height / imgh_face * input_rgb3.rows);
            int location[4];
            location[0] = facetmp.x;
            location[1] = facetmp.y;
            location[2] = facetmp.w;
            location[3] = facetmp.h;
            
            /* 坐标转换 */
            rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);
            MNNRect vis_track;
            vis_track.x = location[0];
            vis_track.y = location[1];
            vis_track.w = location[2];
            vis_track.h = location[3];
            faceResult.push_back(vis_track);
            
            /* 人脸当跟踪显示 */
            //vis_track.score = 100.0;
            //vis_track.handFlag = 3;
            //result.push_back(vis_track);
            
        }
        
        
        clock_gettime(CLOCK_REALTIME, &time2);
        printf("[MNN] time 10 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
        /* 开启跟随 */
        if( command_status.track_status==1 )
        {
            /* 初始帧模版初始化 */
            /* 暂时未考虑旋转情况下进行跟踪目标的选择 */
            if( command_status.initflag==1 )
            {
                /* 图像通道分离 */
                cvMat32sMat(input_rgb, framer, frameg, frameb, framergb);
                
                /* 跟踪参数设置 */
                pvTrack.setParam(2, 1, 1 ,1 ,1 ,2);

                /* accept接收跟踪初始坐标 */
                result1[0] = xmin;
                result1[1] = ymin;
                result1[2] = width;
                result1[3] = height;
                
                /* init location rotation */
                initTrack_rotation(result1, input_rgb3.cols, input_rgb3.rows, 640, 480);
                
                printf("[mnn2] is result1 (%f, %f, %f, %f)\n",result1[0],result1[1],result1[2],result1[3] );
                
                /* 初始帧检测判断 */
                if( faces.size()<1 )  // 没有人脸则进行重新检测，确定检测生效
                {
                    detFace(input_rgb, path, faces, landmarkBoxResult,1);
                    printf("[mnn2] detect face (%d, %d, %d, %d, %d)\n", faces.size(), faces[0].x, faces[0].y, faces[0].width, faces[0].height );
                }
                
                /* 检测和跟踪融合判断 */
                float initTmp[4];
                initTmp[0] = result1[0];
                initTmp[1] = result1[1];
                initTmp[2] = result1[2];
                initTmp[3] = result1[3];
                fusionInf fusionInformation;
                fusionInformation = fusionIou(faces, initTmp, input_rgb.cols, input_rgb.rows);
   
                /* 检测有效，更新坐标 */
                if( fusionInformation.iou>0.0001 )
                {
                    cv::Rect facetmp;
                    facetmp.x = (int)((float)faces[fusionInformation.faceid].x/320*input_rgb.cols);
                    facetmp.y = (int)((float)faces[fusionInformation.faceid].y/240*input_rgb.rows);
                    facetmp.width = (int)((float)faces[fusionInformation.faceid].width/320*input_rgb.cols);
                    facetmp.height = (int)((float)faces[fusionInformation.faceid].height/240*input_rgb.rows);
  
                    facetmp.x = facetmp.x - facetmp.width * 0.0;
                    facetmp.y = facetmp.y - facetmp.height * 0.0;
                    facetmp.width = facetmp.width * 1.0;
                    facetmp.height = facetmp.height *1.0;

                    result1[0] = facetmp.x;
                    result1[1] = facetmp.y;
                    result1[2] = facetmp.width;
                    result1[3] = facetmp.height;
                    result1[4] = 100;
                    result1[5] = 0;
                    
                    printf("[mnn2] is result1 fusion (%f, %f, %f, %f)\n",result1[0],result1[1],result1[2],result1[3] );
                }
                
                /* init template */
                pvTrack.init(result1, framergb, framer.rows, framer.cols, initflag_update);

                /*  kalman初始化  */
                kalman1_init(&state1, result1[0], 5e1);
                kalman1_init(&state2, result1[1], 5e1);
                kalman1_init(&state3, result1[2], 5e1);
                kalman1_init(&state4, result1[3], 5e1);
                
                kalman1_init(&state11, result1[0], 5e1);
                kalman1_init(&state22, result1[1], 5e1);
                kalman1_init(&state33, result1[2], 5e1);
                kalman1_init(&state44, result1[3], 5e1);
                
                /* 发送app滤波坐标 */
                resultApp[0] = result1[0];
                resultApp[1] = result1[1];
                resultApp[2] = result1[2];
                resultApp[3] = result1[3];
                
                resultApp_hist[0] = result1[0];
                resultApp_hist[1] = result1[1];
                resultApp_hist[2] = result1[2];
                resultApp_hist[3] = result1[3];
                
                /* 状态设置 */
                if( command_status.initflag )
                {
                    fusionDetect_delay = 5;
                }else{
                    //printf("fusionDetect_delay is %d\n", fusionDetect_delay);
                    fusionDetect_delay--;
                    if( fusionDetect_delay<-1 ) fusionDetect_delay = -1;
                }
                    
                if( fusionInformation.faceid==-1 && fusionDetect_delay>-1 )
                {
                    command_status.initflag = 2; // 检测失效，进入堵塞等待5帧
                }else{
                    command_status.initflag = 0; // 完成初始化跟踪
                }
                initflag_update = 0; // 初始化内存完毕，不再进行初始化
                detectId = 0;  // 图像帧数或者是跟踪帧数
                
                /* 跟踪状态判断 */
                trackState.trackId = 0;      // 跟踪帧数
                trackState.losingStart = 0;  // 初始丢失帧id
                trackState.losingEnd = 0;    // 结束丢失帧id
                trackState.objectLabel = 0;  // 跟踪目标类别
                trackState.score = 100;      // 跟踪得分
                trackState.detectFlag = 0;   // 跟踪检测标志
                trackState.losingSum = 0;    // 跟踪丢失帧数
                trackState.trackResult[0] = result1[0];
                trackState.trackResult[1] = result1[1];
                trackState.trackResult[2] = result1[2];
                trackState.trackResult[3] = result1[3];
                trackState.losingFlag = 0;  // 跟踪丢失标志位
                
                /* save image */
                //saveimg(path1, input_rgb1);
            }
            else
            {
                /* 当前跟踪的帧数 */
                trackState.trackId++;
                
                /* 图像转换 */
                cvMat32sMat(input_rgb, framer, frameg, frameb, framergb);
                
                clock_gettime(CLOCK_REALTIME, &time2);
                printf("[MNN] time 2 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
                
                /* 判断跟踪过程中是否存在旋转 */
                if( angleUpdateFlag==1 && rotateTrackFlag == 1 )
                {
                    /* 坐标归一化 */
                    int location[4];
                    if( (command_hist_status.angleCamera==180 || command_hist_status.angleCamera==0) || (img_w*1.0/img_h>1.5) )
                    {
                        location[0] = result1[0] / 640 * input_rgb.cols;
                        location[1] = result1[1] / 480 * input_rgb.rows;
                        location[2] = result1[2] / 640 * input_rgb.cols;
                        location[3] = result1[3] / 480 * input_rgb.rows;
                    }else{
                        location[0] = result1[0] / 640 * input_rgb3.cols;
                        location[1] = result1[1] / 480 * input_rgb3.rows;
                        location[2] = result1[2] / 640 * input_rgb3.cols;
                        location[3] = result1[3] / 480 * input_rgb3.rows;
                    }
                    
                    /* 转换为竖屏坐标下的像素坐标 */
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_hist_status.angleCamera, command_hist_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 当前角度下的坐标转换 */
                    float location_tmp[4];
                    location_tmp[0] = location[0];
                    location_tmp[1] = location[1];
                    location_tmp[2] = location[2];
                    location_tmp[3] = location[3];
                    currentAngleLocation(location_tmp, result1, command_status.angleCamera, command_status.orientationCamera, input_rgb3.cols, input_rgb3.rows, input_rgb.cols, input_rgb.rows, img_w, img_h);
                    
                    /* 显示跟踪框 */
                    cv::Rect vis_box;
                    vis_box.x = result1[0];
                    vis_box.y = result1[1];
                    vis_box.width = result1[2];
                    vis_box.height = result1[3];
                    cv::rectangle(input_rgb, vis_box, cv::Scalar(255, 0, 0, 255));
                    
                    /* 跟踪初始化 */
                    //pvTrack.init(result1, framergb, framer.rows, framer.cols, 1);
                    
                    /* kalman初始化 */
                    kalman1_init(&state1, result1[0], 5e1);
                    kalman1_init(&state2, result1[1], 5e1);
                    kalman1_init(&state3, result1[2], 5e1);
                    kalman1_init(&state4, result1[3], 5e1);
                    
                    kalman1_init(&state11, result1[0], 5e1);
                    kalman1_init(&state22, result1[1], 5e1);
                    kalman1_init(&state33, result1[2], 5e1);
                    kalman1_init(&state44, result1[3], 5e1);
                    
                    /* 发送app滤波坐标 */
                    resultApp[0] = result1[0];
                    resultApp[1] = result1[1];
                    resultApp[2] = result1[2];
                    resultApp[3] = result1[3];
                    
                    resultApp_hist[0] = result1[0];
                    resultApp_hist[1] = result1[1];
                    resultApp_hist[2] = result1[2];
                    resultApp_hist[3] = result1[3];
                    
                    
                }else{  // 旋转情况下不进行跟踪更新
                    /* 跟踪坐标估计 */
                    
                    clock_gettime(CLOCK_REALTIME, &time2);
                    printf("[MNN] time 3 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
                    pvTrack.update(1, framergb, result1);
                    
                    clock_gettime(CLOCK_REALTIME, &time2);
                    printf("[MNN] time 4 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
                    
                    printf("[MNN_object]  is %f, %f, %f, %f\n", result1[0], result1[1], result1[2], result1[3]);
                }
            }
            
            /* 依据检测结果进行跟踪重叠判断 */
            int faceid = -1;  // 检测和跟踪融合的有效人脸id
            int ii = 0;       // 人脸的id
            float faceIouMax = 0.2;  // 检测和跟踪的融合参数
            if( detectId<10 ){
                faceIouMax = 0.1;
            }else{
                faceIouMax = 0.2;
            }
            for(cv::Rect face: faces)
            {
                cv::Rect tracktmp;
                tracktmp.x = (int)result1[0];
                tracktmp.y = (int)result1[1];
                tracktmp.width = (int)result1[2];
                tracktmp.height = (int)result1[3];

                cv:Rect facetmp;
                facetmp.x = (int)((float)face.x/320*input_rgb.cols);
                facetmp.y = (int)((float)face.y/240*input_rgb.rows);
                facetmp.width = (int)((float)face.width/320*input_rgb.cols);
                facetmp.height = (int)((float)face.height/240*input_rgb.rows);
                
                float faceIou = iouFaceTrack(facetmp, tracktmp);
                if( faceIou>faceIouMax )
                {
                    faceIouMax = faceIou;
                    faceid = ii;
                }
                ii++;
            }
            
            /* 检测和跟踪的融合 */
            if( faceid!=-1 && (detectId%10==0 || detectId<200) )
            {
                detectId++;  // 图像当前帧id
                cv::Rect facetmp;
                facetmp.x = (int)((float)faces[faceid].x/320*input_rgb.cols);
                facetmp.y = (int)((float)faces[faceid].y/240*input_rgb.rows);
                facetmp.width = (int)((float)faces[faceid].width/320*input_rgb.cols);
                facetmp.height = (int)((float)faces[faceid].height/240*input_rgb.rows);
 
                facetmp.x = facetmp.x - facetmp.width * 0.05;
                facetmp.y = facetmp.y - facetmp.height * 0.05;
                facetmp.width = facetmp.width * 1.1;
                facetmp.height = facetmp.height *1.1;
     
                /* 算法底层跟踪和检测直接融合，直接作为算法的下一帧的输入信息 */
                result1[0] = kalman1_filter1(&state1, result1[0], (float)facetmp.x);
                result1[1] = kalman1_filter1(&state2, result1[1], (float)facetmp.y);
                result1[2] = kalman1_filter1(&state3, result1[2], (float)facetmp.width);
                result1[3] = kalman1_filter1(&state4, result1[3], (float)facetmp.height);
                
                /* 跟踪结果不滤波，直接和检测结果进行融合 */
                resultApp[0] = result1[0];
                resultApp[1] = result1[1];
                resultApp[2] = result1[2];
                resultApp[3] = result1[3];
                
                /* 对连续帧跟踪输出结果进行滤波处理 */
                resultApp[0] = kalman1_filter2(&state11, resultApp_hist[0], resultApp[0],abs((float)(resultApp_hist[0]-resultApp[0])) );
                resultApp[1] = kalman1_filter2(&state22, resultApp_hist[1], resultApp[1],abs((float)(resultApp_hist[1]-resultApp[1])));
                resultApp[2] = kalman1_filter2(&state33, resultApp_hist[2], resultApp[2],abs((float)(resultApp_hist[2]-resultApp[2])));
                resultApp[3] = kalman1_filter2(&state44, resultApp_hist[3], resultApp[3],abs((float)(resultApp_hist[3]-resultApp[3])));
                
                /* app历史显示信息 */
                resultApp_hist[0] = resultApp[0];
                resultApp_hist[1] = resultApp[1];
                resultApp_hist[2] = resultApp[2];
                resultApp_hist[3] = resultApp[3];

                /* 更新跟踪模版，直接用检测和跟踪结果更新，而非发送app数值 */
                pvTrack.updateTemplateP(1, framergb, result1, 1);
                
                /* 有检测，可更新跟踪目标类别 */
                if( trackState.trackId<60 )
                {
                    trackState.objectLabel = 1;
                }
                /* 是否检测有效，若非人则不进行检测 */
                if( trackState.objectLabel==1 )
                {
                    trackState.detectFlag = 1;
                }
            }else{
                /* 无检测更新直接用下 */
                detectId++;
                /* 无人脸情况下也需要更新kalman状态 */
                kalman1_filter1(&state1, result1[0], result1[0]);
                kalman1_filter1(&state2, result1[1], result1[1]);
                kalman1_filter1(&state3, result1[2], result1[2]);
                kalman1_filter1(&state4, result1[3], result1[3]);
                
                /* 利用历史帧信息更新app显示信息 */
                resultApp[0] = kalman1_filter2(&state11, resultApp_hist[0], result1[0], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[1] = kalman1_filter2(&state22, resultApp_hist[1], result1[1], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[2] = kalman1_filter2(&state33, resultApp_hist[2], result1[2], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[3] = kalman1_filter2(&state44, resultApp_hist[3], result1[3], abs((float)(resultApp_hist[0]-result1[0])) );
                
                /* 更新app历史显示信息 */
                resultApp_hist[0] = resultApp[0];
                resultApp_hist[1] = resultApp[1];
                resultApp_hist[2] = resultApp[2];
                resultApp_hist[3] = resultApp[3];
                
                /* 更新跟踪模版 */
                pvTrack.updateTemplateP(1, framergb, result1, 0);
                
                /* 当前检测状态 */
                trackState.detectFlag = 0;  // 无检测目标
            }
            
            /* 跟踪丢失状态的判断 */
            if(1)
            {
                /* 先判断指定目标的跟随丢失判断 */
                if( result1[4]<60 )
                {
                    trackState.losingSum++;
                    if( trackState.losingSum==1 )
                    {
                        trackState.losingStart = trackState.trackId;
                        trackState.losingEnd = trackState.trackId;
                    }
                    else{
                        trackState.losingEnd = trackState.trackId;  // 跟踪丢失的帧数
                    }
                }
                else{
                    trackState.losingSum = 0;
                    trackState.losingStart = 0;
                    trackState.losingEnd = 0;
                    
                    /* 跟踪未丢失，更新历史状态 */
                    trackState.losingFlag = 0;
                    trackState.score = result1[4];
                    trackState.trackResult[0] = result1[0];
                    trackState.trackResult[1] = result1[1];
                    trackState.trackResult[2] = result1[2];
                    trackState.trackResult[3] = result1[3];
                }
                
                /* 有检测情况下状态全部清0 */
                if( trackState.detectFlag == 1 )
                {
                    trackState.losingSum = 0;
                    trackState.losingStart = 0;
                    trackState.losingEnd = 0;
                }
                
                printf("[MNN]_____losing is %f, %d， %d\n", result1[4], trackState.losingStart,trackState.losingEnd);
                
                /* 判断当前跟踪状态：正常跟随、跟踪搜寻状态、跟踪丢失状态 */
                // 跟踪连续丢失50帧则退出跟随
                if( (trackState.losingEnd-trackState.losingStart)>60 )
                {
                    trackState.score = 40;
                    trackState.losingFlag = 1;
                    command_status.track_status = 0; // 跟踪丢失退出跟随
                }else if( (trackState.losingEnd-trackState.losingStart)>15 ){
                    trackState.score = 55;
                    trackState.losingFlag = 1;
                }else{
                    //trackState.score = result1[4];
                    trackState.score = 100;
                    trackState.losingFlag = 0;
                }
            }else
            {
                // 跟踪未丢失状态
                trackState.losingFlag = 0;
                trackState.score = result1[4];
                trackState.trackResult[0] = result1[0];
                trackState.trackResult[1] = result1[1];
                trackState.trackResult[2] = result1[2];
                trackState.trackResult[3] = result1[3];
            }
            
            /* 如果跟踪得分很低，防止跟踪乱跑，跟踪结果不变(包含底层运行输入和app输入) */
            if( trackState.losingFlag==-1 )
            {
                result1[0] = trackState.trackResult[0];
                result1[1] = trackState.trackResult[1];
                result1[2] = trackState.trackResult[2];
                result1[3] = trackState.trackResult[3];
                
                resultApp[0] = trackState.trackResult[0];
                resultApp[1] = trackState.trackResult[1];
                resultApp[2] = trackState.trackResult[2];
                resultApp[3] = trackState.trackResult[3];
                
                resultApp_hist[0] = trackState.trackResult[0];
                resultApp_hist[1] = trackState.trackResult[1];
                resultApp_hist[2] = trackState.trackResult[2];
                resultApp_hist[3] = trackState.trackResult[3];
            }
            
            /* 发送app跟踪坐标 */
            if(1)
            {
                MNNRect vis_track;
                int location[4];
                location[0] = resultApp_hist[0]/640*input_rgb3.cols;
                location[1] = resultApp_hist[1]/480*input_rgb3.rows;
                location[2] = resultApp_hist[2]/640*input_rgb3.cols;
                location[3] = resultApp_hist[3]/480*input_rgb3.rows;
                
                /* 旋转情况下跟踪坐标的发送 */
                if( angleUpdateFlag==1 && rotateTrackFlag == 1 )
                {
                    /* 旋转完成此状态位置0 */
                    angleUpdateFlag++;
                    
                    /* 转换到竖屏坐标下 */
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 更新当前旋转角度 */
                    command_hist_status.angleCamera = command_status.angleCamera;
                    command_hist_status.orientationCamera = command_status.orientationCamera;
                    
                    /* 前置摄像头 */
                    if( command_hist_status.angleCamera==0 && command_status.orientationCamera==1 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3]/3.0*4.0;
                    }
                    else if( command_hist_status.angleCamera==-90 && command_status.orientationCamera==1 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1];
                        vis_track.w = location[2];
                        vis_track.h = location[3];
                    }
                    else if( command_hist_status.angleCamera==180 && command_status.orientationCamera==1 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3]/3.0*4.0;
                    }
                    else if( command_hist_status.angleCamera==-90 && command_status.orientationCamera==1 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1];
                        vis_track.w = location[2];
                        vis_track.h = location[3];
                    }
                    /* 旋转情况下不发送跟踪坐标那一帧 */
                }
                else
                {
                    printf("[MNN_object] location is %d, %d, %d, %d, (%d, %d)\n", location[0], location[1], location[2], location[3], command_status.angleCamera, command_status.orientationCamera);
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_hist_status.angleCamera, command_hist_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 发送app的坐标 */
                    vis_track.x = location[0];
                    vis_track.y = location[1];
                    vis_track.w = location[2];
                    vis_track.h = location[3];
                    vis_track.score = trackState.score;  // 当前跟踪的得分
                    vis_track.handFlag = handflag;
                    result.push_back(vis_track);
                    
                    printf("[MNN]_____track score is %f\n", trackState.score);
                }
                
                
                printf("[MNN_object] app is %d, %d, %d, %d, (%d, %d)\n", location[0], location[1], location[2], location[3], command_status.angleCamera, command_status.orientationCamera);
                
                /* 发送app的坐标 */
                //vis_track.x = location[0];
                //vis_track.y = location[1];
                //vis_track.w = location[2];
                //vis_track.h = location[3];
                //vis_track.score = 100;
                //vis_track.handFlag = handflag;
                //result.push_back(vis_track);
            }
        } // 跟踪结束
        else{
            /*  无跟踪发送检测结果 */
            MNNRect vis_track;
            vis_track.x = 0;
            vis_track.y = 0;
            vis_track.w = 1;
            vis_track.h = 1;
            vis_track.score = -1;
            vis_track.handFlag = handflag;
            result.push_back(vis_track);
            detectId++;
            
            /* 旋转完成更新历史相机姿态角度 */
            if( angleUpdateFlag==1 && rotateTrackFlag == 1 )
            {
                /* 旋转完成此状态位置0 */
                angleUpdateFlag++;
                
                /* 更新当前旋转角度 */
                command_hist_status.angleCamera = command_status.angleCamera;
                command_hist_status.orientationCamera = command_status.orientationCamera;
            }
    
        }
    }else  /* shoulder track */
    {
        //printf("shoulder detect \n");
        /* 全图手势检测或者跟随手势检测 */
        if( (command_status.track_status!=1) ) // 进入全图手势检
        {
            if( detectId%10==0 || detectId<200  )   // 抽帧进行计算
            {
                shoulderPerson(input_rgb, path, shoulders, 1);
                /* 只有检测则进行全图手势检测 */
                if( command_status.hand_status==1 )
                {
                    for(cv::Rect shoulder: shoulders)
                    {
                        /* 获取手势检测区域 */
                        float shoulder_w = 320.0;
                        float shoulder_h = 256.0;
                        cv::Rect handbox;
                        handbox.x = shoulder.x / shoulder_w * input_rgb.cols;
                        handbox.y = shoulder.y / shoulder_h * input_rgb.rows;
                        handbox.width = shoulder.width / shoulder_w * input_rgb.cols;
                        handbox.height = shoulder.height / shoulder_h * input_rgb.rows;
                        //printf("handbox is %d, %d, %d, %d\n", handbox.x, handbox.y, handbox.width, handbox.height);
                        
                        handbox.x = handbox.x - handbox.width*1;
                        handbox.y = handbox.y - handbox.height*1;
                        handbox.width = handbox.width*3;
                        handbox.height = handbox.height*3;
     
                        if( handbox.x<1 ) handbox.x = 2;
                        if( handbox.y<1 ) handbox.y = 2;
                        if( (handbox.x+handbox.width)>input_rgb.cols-2 ) handbox.width = input_rgb.cols-handbox.x;
                        if( (handbox.y+handbox.height)>input_rgb.rows-2 ) handbox.height = input_rgb.rows-handbox.y;
                        if( handbox.width<0 || handbox.height<0 ) continue;

                        /* 图像填充，保证检测图像的畸变问题 */
                        int top, bottom, left, right;
                        cv::Mat handimg1 = input_rgb(handbox);
                        //printf("handbox is %d, %d, %d, %d\n", handbox.x, handbox.y, handbox.width, handbox.height);
                        if( handimg1.cols>handimg1.rows )
                        {
                            top = 0;
                            bottom = handimg1.cols-handimg1.rows;
                            left = 0;
                            right =0;
                        }else
                        {
                            top = 0;
                            bottom = 0;
                            left = 0;
                            right = handimg1.rows - handimg1.cols;
                        }
                        cv::Mat handing;
                        copyMakeBorder(handimg1, handing, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(0, 0, 0));
                        
                        /* 手势检测 */
                        int hand_imgw = handing.cols;
                        int hand_imgh = handing.rows;
                        std::vector<cv::Rect> hands;
                        detectHand(handing, path, hands);
                        
                        /* 抠图进行手势识别 */
                        for(cv::Rect hand: hands)
                        {
                            //printf("hand 1 \n");
                            hand.x = fmax((float)hand.x/176*hand_imgw+handbox.x, 2);
                            hand.y = fmax((float)hand.y/176*hand_imgh+handbox.y, 2);
                            hand.width = fmin((float)hand.width/176*hand_imgw, hand_imgw);
                            hand.height = fmin((float)hand.height/176*hand_imgh, hand_imgh);
                            cv::Rect vis_hand;
                            vis_hand.x = fmax(hand.x-hand.width*0.1, 2);
                            vis_hand.y = fmax(hand.y-hand.height*0.1, 2);
                            vis_hand.width = int(fmin(hand.x+hand.width*1.1, input_rgb.cols) - hand.x);
                            vis_hand.height = int(fmin(hand.y+hand.height*1.1, input_rgb.rows) - hand.y);
                            if( vis_hand.width<0 || vis_hand.height<0 ) continue;
                            
                            cv::Mat handimg_r = input_rgb(vis_hand);
                            
                            /* 手势识别 */
                            /* 0 video; 2 picture; 3 group; 4 track; 1 nohand */
                            handflag = recognitionHand(handimg_r, path);
                            //printf("handflag is %d\n", handflag);
                            /* 开启跟随 */
                            if( handflag==3 )
                            {
                               /* 获取人脸坐标 */
                                float face_imgw = 320.0;
                                float face_imgh = 256.0;
                                cv::Rect trackbox;
                                trackbox.x = (int)(shoulder.x/face_imgw*input_rgb3.cols);
                                trackbox.y = (int)(shoulder.y/face_imgh*input_rgb3.rows);
                                trackbox.width = (int)(shoulder.width/face_imgw*input_rgb3.cols);
                                trackbox.height = (int)(shoulder.height/face_imgh*input_rgb3.rows);
                                /* 跟踪参数设置 */
                                xmin = trackbox.x;
                                ymin = trackbox.y;
                                width = trackbox.width;
                                height = trackbox.height;
                                if( command_status.track_status==0 && handdelay>80)  // 开启跟随
                                {
                                    handdelay = 0;
                                    command_status.track_status = 1;
                                    command_status.initflag = 1;
                                }else if( handdelay>80 ){
                                    handdelay = 0;
                                    command_status.track_status = 0;   // 关闭跟随
                                    command_status.initflag = 0;
                                }
                            }
                            if( handflag!= 1 ) break;
                        }
                        if( handflag!= 1 ) break;
                    }
                }
            }
        }
        else if( command_status.track_status==1 ) // 进入跟踪手势检测
        {
            if( detectId%10==0 )   // 抽帧进行计算
            {
                /* 获取头肩局部检测区域，6倍区域 */
                cv::Rect regionBox;
                regionBox.x = result1[0] - result1[2]*2.5;
                regionBox.y = result1[1] - result1[3]*2.5;
                regionBox.width = result1[2]*6;
                regionBox.height = result1[3]*6.0;
                
                /* 越界判断 */
                if( regionBox.x<1 ) regionBox.x = 1;
                if( regionBox.y<1 ) regionBox.y = 1;
                if( (regionBox.x+regionBox.width)>input_rgb.cols-2 )
                    regionBox.width = input_rgb.cols - regionBox.x;
                if( (regionBox.y+regionBox.height)>input_rgb.rows-2 )
                    regionBox.height = input_rgb.rows - regionBox.y;
                if( regionBox.width<0 || regionBox.height<0)
                {
                    regionBox.x = 1;
                    regionBox.y = 1;
                    regionBox.width = 20;
                    regionBox.height = 20;
                }
                
                /* 局部抠图检测 */
                cv::Mat regionBox_img = input_rgb(regionBox);
                
                /* 局部检测 */
                std::vector<cv::Rect> shoulders_tmp;
                shoulderPerson(regionBox_img, path, shoulders_tmp, 2);
                
                /* 局部坐标转换为全局坐标 */
                float regionBox_imgW = 160;
                float regionBox_imgH = 128;
                for( cv::Rect shoulder_tmp: shoulders_tmp )
                {
                    /* 转换为VGA坐标 */
                    shoulder_tmp.x = (float)shoulder_tmp.x / regionBox_imgW * regionBox_img.cols + regionBox.x;
                    shoulder_tmp.y = (float)shoulder_tmp.y / regionBox_imgH * regionBox_img.rows + regionBox.y;
                    shoulder_tmp.width = (float)shoulder_tmp.width / regionBox_imgW * regionBox_img.cols;
                    shoulder_tmp.height = (float)shoulder_tmp.height / regionBox_imgH * regionBox_img.rows;
                    
                    /* 640*480坐标转换为320*256坐标 */
                    shoulder_tmp.x = shoulder_tmp.x * 0.5;
                    shoulder_tmp.y = shoulder_tmp.y * 256.0 / 480.0;
                    shoulder_tmp.width = shoulder_tmp.width * 0.5;
                    shoulder_tmp.height = shoulder_tmp.height * 256.0 / 480.0;
                    shoulders.push_back(shoulder_tmp);
                }
                
                //shoulderPerson(input_rgb, path, shoulders, 2);
                if( command_status.hand_status==1 )
                {
                    /* 获取手势检测区域 */
                    cv::Rect handbox;
                    handbox.x = result1[0] - result1[2]*0.5;
                    handbox.y = result1[1] - result1[3]*0.5;
                    handbox.width = result1[2]*2;
                    handbox.height = result1[3]*2;
                    if( handbox.x<1 ) handbox.x = 2;
                    if( handbox.y<1 ) handbox.y = 2;
                    if( (handbox.x+handbox.width)>input_rgb.cols-2 ) handbox.width = input_rgb.cols-handbox.x;
                    if( (handbox.y+handbox.height)>input_rgb.rows-2 ) handbox.height = input_rgb.rows-handbox.y;
                    if( handbox.width<0 || handbox.height<0 )
                    {
                        handbox.x = 1;
                        handbox.y = 1;
                        handbox.width = 20;
                        handbox.height = 20;
                    }
                        
                    /* 图像填充，保证检测图像的畸变问题 */
                    int top, bottom, left, right;
                    cv::Mat handimg1 = input_rgb(handbox);
                    if( handimg1.cols>handimg1.rows )
                    {
                        top = 0;
                        bottom = handimg1.cols-handimg1.rows;
                        left = 0;
                        right =0;
                    }else
                    {
                        top = 0;
                        bottom = 0;
                        left = 0;
                        right = handimg1.rows - handimg1.cols;
                    }
                    cv::Mat handing;
                    copyMakeBorder(handimg1, handing, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(0, 0, 0));
                        
                    /* 手势检测 */
                    int handw = handing.cols;
                    int handh = handing.rows;
                    std::vector<cv::Rect> hands;
                    detectHand(handing, path, hands);
                        
                    /* 抠图进行手势识别 */
                    for(cv::Rect hand: hands)
                    {
                        hand.x = fmax((float)hand.x/176*handw+handbox.x, 2);
                        hand.y = fmax((float)hand.y/176*handh+handbox.y, 2);
                        hand.width = fmin((float)hand.width/176*handw, handw);
                        hand.height = fmin((float)hand.height/176*handh, handh);
                        cv::Rect vis_hand;
                        vis_hand.x = fmax(hand.x-hand.width*0.1, 2);
                        vis_hand.y = fmax(hand.y-hand.height*0.1, 2);
                        vis_hand.width = int(fmin(hand.x+hand.width*1.1, input_rgb.cols) - hand.x);
                        vis_hand.height = int(fmin(hand.y+hand.height*1.1, input_rgb.rows) - hand.y);
                        if( vis_hand.width<0 || vis_hand.height<0 ) continue;
                        
                        cv::Mat handimg_r = input_rgb(vis_hand);
                        
                        /* 手势识别 */
                        handflag = recognitionHand(handimg_r, path);
                        //printf("handflag is %d\n", handflag);
                        /* 开启跟随 */
                        if( handflag==3 )
                        {
                            if( handdelay>80 ){  // 关闭跟随
                                handdelay = 0;
                                command_status.track_status = 0;
                                command_status.initflag = 0;
                            }
                        }
                        if( handflag!= 1 ) break;
                    }
                }
            }
        }// end shoulder and track hand
        
        /* 发送人脸检测结果 */
        for( cv::Rect shoulder: shoulders )
        {
            float imgw_shoulder = 320.0;
            float imgh_shoulder = 256.0;
            MNNRect shouldertmp;
            shouldertmp.x = (int) (shoulder.x / imgw_shoulder * input_rgb3.cols);
            shouldertmp.y = (int) (shoulder.y / imgh_shoulder * input_rgb3.rows);
            shouldertmp.w = (int) (shoulder.width / imgw_shoulder * input_rgb3.cols);
            shouldertmp.h = (int) (shoulder.height / imgh_shoulder * input_rgb3.rows);
            int location[4];
            location[0] = shouldertmp.x;
            location[1] = shouldertmp.y;
            location[2] = shouldertmp.w;
            location[3] = shouldertmp.h;
            
            /* 坐标转换 */
            rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);
            MNNRect vis_track;
            vis_track.x = location[0];
            vis_track.y = location[1];
            vis_track.w = location[2];
            vis_track.h = location[3];
            faceResult.push_back(vis_track);
        }
        
        printf("-----command_status.track_status is %d\n", command_status.track_status);

        /* 开启跟随 */
        if( command_status.track_status==1 )
        {
            /* 初始帧模版初始化 */
            if( command_status.initflag==1 )
            {
                /* 图像通道分离 */
                cvMat32sMat(input_rgb, framer, frameg, frameb, framergb);
                
                /* 跟踪参数设置 */
                pvTrack.setParam(2, 1, 1 ,1 ,1 ,2);

                /* accept接收跟踪初始坐标 */
                result1[0] = xmin;
                result1[1] = ymin;
                result1[2] = width;
                result1[3] = height;
                
                /* init location rotation */
                initTrack_rotation(result1, input_rgb3.cols, input_rgb3.rows, 640, 480);
                
                /* 初始帧检测判断 */
                if( shoulders.size()<1 )  // 没有人脸则进行重新检测，确定检测生效
                {
                    shoulderPerson(input_rgb, path, shoulders, 1);
                }
                
                /* 检测和跟踪融合判断 */
                float initTmp[4];
                initTmp[0] = xmin;
                initTmp[1] = ymin;
                initTmp[2] = width;
                initTmp[3] = height;
                fusionInf fusionInformation;
                fusionInformation = fusionIou(shoulders, initTmp, input_rgb.cols, input_rgb.rows);
                
                /* 检测有效，更新坐标 */
                if( fusionInformation.iou>0.0001 )
                {
                    float shoulder_w = 320.0;
                    float shoulder_h = 256.0;
                    cv::Rect facetmp;
                    facetmp.x = shoulders[fusionInformation.faceid].x / shoulder_w * input_rgb.cols;
                    facetmp.y = shoulders[fusionInformation.faceid].y / shoulder_h * input_rgb.rows;
                    facetmp.width = shoulders[fusionInformation.faceid].width / shoulder_w * input_rgb.cols;
                    facetmp.height = shoulders[fusionInformation.faceid].height / shoulder_h * input_rgb.rows ;

                    facetmp.x = facetmp.x;
                    facetmp.y = facetmp.y;
                    facetmp.width = facetmp.width;
                    facetmp.height = facetmp.height;
                    
                    result1[0] = facetmp.x;
                    result1[1] = facetmp.y;
                    result1[2] = facetmp.width;
                    result1[3] = facetmp.height;
                    result1[4] = 100;
                    result1[5] = 0;
                }
                
                /* init template */
                pvTrack.init(result1, framergb, framer.rows, framer.cols, initflag_update);
                
                /*  kalman初始化 */
                kalman1_init(&state1, result1[0], 5e1);
                kalman1_init(&state2, result1[1], 5e1);
                kalman1_init(&state3, result1[2], 5e1);
                kalman1_init(&state4, result1[3], 5e1);
                
                kalman1_init(&state11, result1[0], 5e1);
                kalman1_init(&state22, result1[1], 5e1);
                kalman1_init(&state33, result1[2], 5e1);
                kalman1_init(&state44, result1[3], 5e1);
                
                /* 发送app坐标 */
                resultApp[0] = result1[0];
                resultApp[1] = result1[1];
                resultApp[2] = result1[2];
                resultApp[3] = result1[3];
                
                resultApp_hist[0] = result1[0];
                resultApp_hist[1] = result1[1];
                resultApp_hist[2] = result1[2];
                resultApp_hist[3] = result1[3];
                
                /* 状态设置 */
                if( command_status.initflag )
                {
                    fusionDetect_delay = 5;
                }else{
                    //printf("fusionDetect_delay is %d\n", fusionDetect_delay);
                    fusionDetect_delay--;
                    if( fusionDetect_delay<-1 ) fusionDetect_delay = -1;
                }
                    
                if( fusionInformation.faceid==-1 && fusionDetect_delay>-1 )
                {
                    command_status.initflag = 2; // 检测失效，进入堵塞等待5帧
                }else{
                    command_status.initflag = 0; // 完成初始化跟踪
                }
                initflag_update = 0; // 初始化内存完毕，不再进行初始化
                detectId = 0;  // 图像帧数或者是跟踪帧数
                
                /* 跟踪丢失状态判断 */
                trackState.trackId = 0;
                trackState.losingStart = 0;
                trackState.losingEnd = 0;
                trackState.objectLabel = 0;
                trackState.score = 100;
                trackState.detectFlag = 0;
                trackState.losingSum = 0;
                trackState.trackResult[0] = result1[0];
                trackState.trackResult[1] = result1[1];
                trackState.trackResult[2] = result1[2];
                trackState.trackResult[3] = result1[3];
                trackState.losingFlag = 0;
            }
            else
            {
                /* 当前跟踪的帧数 */
                trackState.trackId++;
                
                /* 图像转换 */
                cvMat32sMat(input_rgb, framer, frameg, frameb, framergb);
                
                /* 判断跟踪过程中是否存在旋转 */
                if( angleUpdateFlag==1 && rotateTrackFlag == 1)
                {
                    /* 坐标归一化 */
                    int location[4];
                    location[0] = result1[0] / 640 * input_rgb3.cols;
                    location[1] = result1[1] / 480 * input_rgb3.rows;
                    location[2] = result1[2] / 640 * input_rgb3.cols;
                    location[3] = result1[3] / 480 * input_rgb3.rows;
                    
                    /* 转换为竖屏坐标下的像素坐标 */
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_hist_status.angleCamera, command_hist_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 当前角度下的坐标转换 */
                    float location_tmp[4];
                    location_tmp[0] = location[0];
                    location_tmp[1] = location[1];
                    location_tmp[2] = location[2];
                    location_tmp[3] = location[3];
                    currentAngleLocation(location_tmp, result1, command_status.angleCamera, command_status.orientationCamera, input_rgb3.cols, input_rgb3.rows, input_rgb.cols, input_rgb.rows, img_w, img_h);
                    
                    /* 显示跟踪框 */
                    cv::Rect vis_box;
                    vis_box.x = result1[0];
                    vis_box.y = result1[1];
                    vis_box.width = result1[2];
                    vis_box.height = result1[3];
                    
                    /* 跟踪初始化 */
                    //pvTrack.init(result1, framergb, framer.rows, framer.cols, 1);
                    
                    /* kalman初始化 */
                    kalman1_init(&state1, result1[0], 5e1);
                    kalman1_init(&state2, result1[1], 5e1);
                    kalman1_init(&state3, result1[2], 5e1);
                    kalman1_init(&state4, result1[3], 5e1);
                    
                    kalman1_init(&state11, result1[0], 5e1);
                    kalman1_init(&state22, result1[1], 5e1);
                    kalman1_init(&state33, result1[2], 5e1);
                    kalman1_init(&state44, result1[3], 5e1);
                    
                    /* 发送app滤波坐标 */
                    resultApp[0] = result1[0];
                    resultApp[1] = result1[1];
                    resultApp[2] = result1[2];
                    resultApp[3] = result1[3];
                    
                    resultApp_hist[0] = result1[0];
                    resultApp_hist[1] = result1[1];
                    resultApp_hist[2] = result1[2];
                    resultApp_hist[3] = result1[3];
                    
                }else  /* 旋转情况下不更新跟踪坐标 */
                {
                    /* 跟踪坐标估计 */
                    pvTrack.update(1, framergb, result1);
                }
            }
            
            
            /* 依据检测结果进行跟踪重叠判断 */
            int faceid = -1;  // 检测和跟踪融合的有效人脸id
            int ii = 0;       // 人脸的id
            float faceIouMax = 0.2;  // 检测和跟踪的融合参数
            if( detectId<10 ){
                faceIouMax = 0.1;
            }else{
                faceIouMax = 0.2;
            }
            for(cv::Rect shoulder: shoulders)
            {
                int shoulder_w = 320.0;
                int shoulder_h = 256.0;
                cv::Rect tracktmp;
                tracktmp.x = (int)result1[0];
                tracktmp.y = (int)result1[1];
                tracktmp.width = (int)result1[2];
                tracktmp.height = (int)result1[3];
                
                cv::Rect facetmp;
                facetmp.x = shoulder.x / shoulder_w * input_rgb.cols;
                facetmp.y = shoulder.y / shoulder_h * input_rgb.rows;
                facetmp.width = shoulder.width / shoulder_w * input_rgb.cols;
                facetmp.height = shoulder.height / shoulder_h * input_rgb.rows;

                float faceIou = iouFaceTrack(facetmp, tracktmp);
                if( faceIou>faceIouMax )
                {
                    faceIouMax = faceIou;
                    faceid = ii;
                }
                ii++;
            }
            
            /* 检测和跟踪的融合 */
            if( faceid!=-1 && (detectId%10==0 || detectId<200) )
            {
                int shoulder_w = 320.0;
                int shoulder_h = 256.0;
                detectId++;  // 图像当前帧id
                cv::Rect facetmp;
                facetmp.x = shoulders[faceid].x / shoulder_w * input_rgb.cols;
                facetmp.y = shoulders[faceid].y / shoulder_h * input_rgb.rows;
                facetmp.width = shoulders[faceid].width / shoulder_w * input_rgb.cols;
                facetmp.height = shoulders[faceid].height / shoulder_h * input_rgb.rows;
                
                facetmp.x = facetmp.x;
                facetmp.y = facetmp.y;
                facetmp.width = facetmp.width;
                facetmp.height = facetmp.height;
                
                result1[0] = kalman1_filter1(&state1, result1[0], (float)facetmp.x);
                result1[1] = kalman1_filter1(&state2, result1[1], (float)facetmp.y);
                result1[2] = kalman1_filter1(&state3, result1[2], (float)facetmp.width);
                result1[3] = kalman1_filter1(&state4, result1[3], (float)facetmp.height);
                
                /* 跟踪结果不滤波，直接和检测结果进行融合 */
                resultApp[0] = result1[0];
                resultApp[1] = result1[1];
                resultApp[2] = result1[2];
                resultApp[3] = result1[3];
                
                /* 对连续帧跟踪输出结果进行滤波处理 */
                resultApp[0] = kalman1_filter2(&state11, resultApp_hist[0], resultApp[0],abs((float)(resultApp_hist[0]-resultApp[0])) );
                resultApp[1] = kalman1_filter2(&state22, resultApp_hist[1], resultApp[1],abs((float)(resultApp_hist[1]-resultApp[1])));
                resultApp[2] = kalman1_filter2(&state33, resultApp_hist[2], resultApp[2],abs((float)(resultApp_hist[2]-resultApp[2])));
                resultApp[3] = kalman1_filter2(&state44, resultApp_hist[3], resultApp[3],abs((float)(resultApp_hist[3]-resultApp[3])));
                
                /* app历史显示信息 */
                resultApp_hist[0] = resultApp[0];
                resultApp_hist[1] = resultApp[1];
                resultApp_hist[2] = resultApp[2];
                resultApp_hist[3] = resultApp[3];
                
                /* 更新跟踪模版 */
                pvTrack.updateTemplateP(1, framergb, result1, 1);
                /* 有检测，更新跟踪目标类别 */
                if( trackState.trackId<60 )
                {
                    trackState.objectLabel = 1;
                }
                /* 是否检测有效，若非人则不进行检测 */
                if( trackState.objectLabel==1 )
                {
                    trackState.detectFlag = 1;
                }
            }else
            {
                detectId++;
                /* 无人脸情况下也需要更新kalman状态 */
                kalman1_filter1(&state1, result1[0], result1[0]);
                kalman1_filter1(&state2, result1[1], result1[1]);
                kalman1_filter1(&state3, result1[2], result1[2]);
                kalman1_filter1(&state4, result1[3], result1[3]);
                
                /* 利用历史帧信息更新app显示信息 */
                resultApp[0] = kalman1_filter2(&state11, resultApp_hist[0], result1[0], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[1] = kalman1_filter2(&state22, resultApp_hist[1], result1[1], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[2] = kalman1_filter2(&state33, resultApp_hist[2], result1[2], abs((float)(resultApp_hist[0]-result1[0])) );
                resultApp[3] = kalman1_filter2(&state44, resultApp_hist[3], result1[3], abs((float)(resultApp_hist[0]-result1[0])) );
                
                /* 更新app历史显示信息 */
                resultApp_hist[0] = resultApp[0];
                resultApp_hist[1] = resultApp[1];
                resultApp_hist[2] = resultApp[2];
                resultApp_hist[3] = resultApp[3];
                
                /* 更新跟踪模版 */
                pvTrack.updateTemplateP(1, framergb, result1, 0);
                
                /* 无检测融合，更新当前跟踪的检测状态 */
                trackState.detectFlag = 0;
            }
            
            /* 跟踪丢失状态的判断 */
            if(1)
            {
                /* 先判断指定目标的跟随丢失判断 */
                if( result1[4]<65 )
                {
                    trackState.losingSum++;
                    if( trackState.losingSum==1 )
                    {
                        trackState.losingStart = trackState.trackId;
                        trackState.losingEnd = trackState.trackId;
                    }
                    else{
                        trackState.losingEnd = trackState.trackId;  // 跟踪丢失的帧数
                    }
                }
                else{
                    trackState.losingSum = 0;
                    trackState.losingStart = 0;
                    trackState.losingEnd = 0;
                    
                    /* 跟踪未丢失，更新历史状态 */
                    trackState.losingFlag = 0;
                    trackState.score = result1[4];
                }
                
                /* 有检测情况下状态全部清0 */
                if( trackState.detectFlag == 1 )
                {
                    trackState.losingSum = 0;
                    trackState.losingStart = 0;
                    trackState.losingEnd = 0;
                }
                
                printf("[MNN]_____losing is %f, %d， %d\n", result1[4], trackState.losingStart,trackState.losingEnd);
                
                /* 判断当前跟踪状态：正常跟随、跟踪搜寻状态、跟踪丢失状态 */
                // 跟踪连续丢失50帧则退出跟随
                if( (trackState.losingEnd-trackState.losingStart)>60 )
                {
                    trackState.score = 40;
                    trackState.losingFlag = 1;
                    command_status.track_status = 0; // 跟踪丢失退出跟随
                }else if( (trackState.losingEnd-trackState.losingStart)>15 ){
                    trackState.score = 55;
                    trackState.losingFlag = 1;
                }else{
                    //trackState.score = result1[4];
                    trackState.score = 100;
                    trackState.losingFlag = 0;
                }
            }else
            {
                // 跟踪未丢失状态
                trackState.losingFlag = 0;
                trackState.score = result1[4];
            }
            
            if(1)
            {
                /* 发送app跟踪坐标 */
                int location[4];
                MNNRect vis_track;
                location[0] = resultApp_hist[0]/640*input_rgb3.cols;
                location[1] = resultApp_hist[1]/480*input_rgb3.rows;
                location[2] = resultApp_hist[2]/640*input_rgb3.cols;
                location[3] = resultApp_hist[3]/480*input_rgb3.rows;

                /* 旋转情况下跟踪坐标的发送 */
                if( angleUpdateFlag==1 && rotateTrackFlag == 1)
                {
                    angleUpdateFlag++;
                    
                    /* 转换到竖屏坐标下 */
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 更新当前旋转角度 */
                    command_hist_status.angleCamera = command_status.angleCamera;
                    command_hist_status.orientationCamera = command_status.orientationCamera;
                    
                    /* 后置摄像头 */
                    if( command_hist_status.angleCamera==-90 && command_status.orientationCamera==0 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3]/3.0*4.0;
                    }
                    else if( command_hist_status,angleCamera==0 && command_status.orientationCamera==0 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3];
                    }
                    else if( command_hist_status.angleCamera==180 && command_status.orientationCamera==0 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3];
                    }
                    else if( command_hist_status.angleCamera==90 && command_status.orientationCamera==0 )
                    {
                        vis_track.x = location[0];
                        vis_track.y = location[1]/3.0*4.0;
                        vis_track.w = location[2];
                        vis_track.h = location[3]/3.0*4.0;
                    }
                    /* 旋转情况下不发送跟踪坐标 */
                    // to do
                }
                else
                {
                    rotationLocation(location, input_rgb3.cols, input_rgb3.rows, frame.cols-frame.rows, command_hist_status.angleCamera, command_hist_status.orientationCamera, 0, img_w, img_h);
                    
                    /* 发送app坐标 */
                    vis_track.x = location[0];
                    vis_track.y = location[1];
                    vis_track.w = location[2];
                    vis_track.h = location[3];
                    vis_track.score = trackState.score;
                    vis_track.handFlag = handflag;
                    
                    /* 如果跟踪得分很低，防止跟踪乱跑，跟踪结果不变(包含底层运行输入和app输入) */
                    if( trackState.losingFlag==1 )
                    {
                        vis_track.x = trackState.trackResult[0];
                        vis_track.y = trackState.trackResult[1];
                        vis_track.w = trackState.trackResult[2];
                        vis_track.h = trackState.trackResult[3];
                    }else{
                        trackState.trackResult[0] = location[0];
                        trackState.trackResult[1] = location[1];
                        trackState.trackResult[2] = location[2];
                        trackState.trackResult[3] = location[3];
                    }
                    
                    result.push_back(vis_track);
                    
                    printf("[MNN]_____track score is %f\n", trackState.score);
                }
            }
            
            printf("track result is %f, %f, %f, %f\n", result1[0], result1[1], result1[2], result1[3] );
        } // 跟踪结束
        else{
            /*  无跟踪发送检测结果 */
            MNNRect vis_track;
            vis_track.x = 0;
            vis_track.y = 0;
            vis_track.w = 1;
            vis_track.h = 1;
            vis_track.score = -1;
            vis_track.handFlag = handflag;
            result.push_back(vis_track);
            detectId++;  // 图像当前帧id
            
            printf("track result is %f, %f, %f, %f\n", 1.0, 1.0, 1.0, 1.0 );
        }
    }
    
    /* 释放内存 */
    free(framerbuf);
    free(framegbuf);
    free(framebbuf);
    free(framergb);
    
    clock_gettime(CLOCK_REALTIME, &time2);
    printf("[MNN] time 10 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);
    printf("[MNN] frame time is %d\n", time2.tv_nsec/1000000);

    return 0;
}


/* 旋转角度的测试-人脸 */
int faceDetectMainR1801(unsigned char* pixels, int h, int w, const char* path, std::vector<MNNRect> &result)
{
    int img_w = w;
    int img_h = h;
    /* 视频流测试接口 */
    cv::Mat frame(h, w, CV_8UC4, pixels);
    cv::flip(frame, frame, 1);
    cv::Mat input_rgb, input_rgb1, input_rgb2;
    cv::cvtColor(frame, input_rgb1, cv::COLOR_BGRA2RGB);

    /* 相机在右边 */
    rotationImg(input_rgb1, input_rgb2, command_status.angleCamera, command_status.orientationCamera);
    imgPpading(input_rgb2, input_rgb, command_status.angleCamera, command_status.orientationCamera);

    
    //clock_gettime(CLOCK_REALTIME, &time1);
    /* 人脸检测与关键点定位算法 */
    std::vector<cv::Rect> faces;
    std::vector<landmarkFace > landmarkBoxResult;
    detFace(input_rgb, path, faces, landmarkBoxResult, 2);
    
//    clock_gettime(CLOCK_REALTIME, &time2);
//    printf("[MNN] time 10 is %d\n", (time2.tv_sec-time1.tv_sec)*1000 + (time2.tv_nsec-time1.tv_nsec)/1000000);

    /* 发送人脸检测的结果给app */
    for (cv::Rect face: faces)
    {
        float imgw = 320.0;
        float imgh = 240.0;
        MNNRect facetmp;
        facetmp.x = (int) (face.x / imgw * input_rgb.cols);
        facetmp.y = (int) (face.y / imgh * input_rgb.rows);
        facetmp.w = (int) (face.width / imgw * input_rgb.cols);
        facetmp.h = (int) (face.height / imgh * input_rgb.rows);

        int location[4];
        location[0] = facetmp.x;
        location[1] = facetmp.y;
        location[2] = facetmp.w;
        location[3] = facetmp.h;

        //printf("[MNN] 160 \n");

        rotationLocation(location, input_rgb.cols, input_rgb.rows, frame.cols - frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);

        //printf("2 %d, %d, %d, %d\n",location[0],location[1],location[2],location[3] );

        MNNRect vis_faceR;
        vis_faceR.x = location[0];
        vis_faceR.y = location[1];
        vis_faceR.w = location[2];
        vis_faceR.h = location[3];
        vis_faceR.score = 100.0;
        vis_faceR.handFlag = 3;
        result.push_back(vis_faceR);

        /* 坐标旋转 */
        //MNNRect vis_faceR;
        //vis_faceR.x = input_rgb.cols-facetmp.x-facetmp.w;
        //vis_faceR.y = input_rgb.rows-facetmp.y-facetmp.h;
        //vis_faceR.w = facetmp.w;
        //vis_faceR.h = facetmp.h;

        /* 坐标转换 */
        //int tempx = vis_faceR.x;
        //vis_faceR.x = vis_faceR.y;
        //vis_faceR.y = input_rgb.cols - (tempx + vis_faceR.w);

        //int tw = vis_faceR.w;
        //vis_faceR.w = vis_faceR.h;
        //vis_faceR.h = tw;
        // printf("x == %f:,y == %f:,w == %f:,h == %f:,",facetmp.x,facetmp.y,facetmp.w,facetmp.h);
        //result.push_back(vis_faceR);
    }

    return 0;
}

/* 旋转角度的测试-头肩 */
int faceDetectMainR180(unsigned char* pixels, int h, int w, const char* path, std::vector<MNNRect> &result)
{
    int img_w = w;
    int img_h = h;
    /* 视频流测试接口 */
    cv::Mat frame(h, w, CV_8UC4, pixels);
    cv::flip(frame, frame, 1);
    cv::Mat input_rgb, input_rgb1, input_rgb2;
    cv::cvtColor(frame, input_rgb1, cv::COLOR_BGRA2RGB);

    /* 相机在右边 */
    rotationImg(input_rgb1, input_rgb2, command_status.angleCamera, command_status.orientationCamera);
    imgPpading(input_rgb2, input_rgb, command_status.angleCamera, command_status.orientationCamera);

    /* 头肩检测算法 */
    std::vector<cv::Rect> shoulders;
    shoulderPerson(input_rgb,  path,  shoulders, 1);

    /* 发送人脸检测的结果给app */
    for (cv::Rect shoulder: shoulders)
    {
        float imgw = 320.0;
        float imgh = 320.0;
        MNNRect facetmp;
        facetmp.x = (int) (shoulder.x / imgw * input_rgb.cols);
        facetmp.y = (int) (shoulder.y / imgh * input_rgb.rows);
        facetmp.w = (int) (shoulder.width / imgw * input_rgb.cols);
        facetmp.h = (int) (shoulder.height / imgh * input_rgb.rows);

        int location[4];
        location[0] = facetmp.x;
        location[1] = facetmp.y;
        location[2] = facetmp.w;
        location[3] = facetmp.h;

        //printf("shoulder is %d, %d, %d, %d\n", shoulder.x, shoulder.y, shoulder.width, shoulder.height);
        //printf("1 %d, %d, %d, %d\n",location[0],location[1],location[2],location[3] );

        rotationLocation(location, input_rgb.cols, input_rgb.rows, frame.cols - frame.rows, command_status.angleCamera, command_status.orientationCamera, 0, img_w, img_h);

        //printf("2 %d, %d, %d, %d\n",location[0],location[1],location[2],location[3] );

        MNNRect vis_faceR;
        vis_faceR.x = location[0];
        vis_faceR.y = location[1];
        vis_faceR.w = location[2];
        vis_faceR.h = location[3];
        result.push_back(vis_faceR);

        /* 坐标旋转 */
        //MNNRect vis_faceR;
        //vis_faceR.x = input_rgb.cols-facetmp.x-facetmp.w;
        //vis_faceR.y = input_rgb.rows-facetmp.y-facetmp.h;
        //vis_faceR.w = facetmp.w;
        //vis_faceR.h = facetmp.h;

        /* 坐标转换 */
        //int tempx = vis_faceR.x;
        //vis_faceR.x = vis_faceR.y;
        //vis_faceR.y = input_rgb.cols - (tempx + vis_faceR.w);

        //int tw = vis_faceR.w;
        //vis_faceR.w = vis_faceR.h;
        //vis_faceR.h = tw;
        // printf("x == %f:,y == %f:,w == %f:,h == %f:,",facetmp.x,facetmp.y,facetmp.w,facetmp.h);
        //result.push_back(vis_faceR);
    }

    return 0;
}
