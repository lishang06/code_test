#pragma once

#define DLL_PUBLIC __attribute__ ((visibility("default")))

//#ifdef __cplusplus
//extern "C"
//{
//#endif

#include <vector>

typedef struct {
    float x;  /* state */
    float A;  /* x(n)=A*x(n-1)+u(n),u(n)~N(0,q) */
    float H;  /* z(n)=H*x(n)+w(n),w(n)~N(0,r)   */
    float q;  /* process(predict) noise convariance */
    float r;  /* measure noise convariance */
    float p;  /* estimated error convariance */
    float gain;
} kalman1_state;

struct MNNRect{
    float x;         // 跟踪坐标x y w h
    float y;
    float w;
    float h;
    float score;
    int initFlag;    // 跟踪初始化标志位
    int command_track_status;  // 开启跟随和关闭跟随的标志位
    int command_hand_status;   // 开启手势和关闭手势的标志位
    int handFlag;    // 手势识别的类型：0大拇指, 1负样本，2剪刀手，3合影，4五指张开 
};

struct accept_data{
    float x;
    float y;
    float w;
    float h;
    float score;
    int track_status;
    int hand_status;
    int initflag;
};

struct cmd_status{
    int track_status; // 0 stop track; 1 start track  starttrack
    int hand_status; // 0 stop hand; 1 start hand
    int angleCamera; // image rotation angle
    int orientationCamera; //  front camera; rear camera
    int initflag;  // init track flag
    int shift_status;
    
    cmd_status()
    {
        hand_status = 0;
        track_status = 0;
    }
};

struct fusionInf{
    int faceid;
    float iou;
};

struct pvTrackState{
    int trackId;           // 跟踪帧数
    int losingStart;       // 跟踪丢失启始帧
    int losingEnd;         // 跟踪丢失结束帧
    int losingSum = 0;
    int losingFlag = 0;
    int objectLabel;       // 跟踪目标类别，0目标，1人脸，2头肩
    float score;           // 当前跟踪得分
    int detectFlag = 0;
    float trackResult[4];
};

//DLL_PUBLIC int deal(int* pixels, int h, int w, const char* path);
extern void kalman1_init(kalman1_state *state, float init_x, float init_p);
float kalman1_filter1(kalman1_state *state, float z_measure, int diff);


int acceptData(std::vector<accept_data> &initbox, int h, int w);
int acceptControlParam(std::vector<accept_data> &initbox);
void getCameraParam(int angle, int orientation);

int faceDetectMainR180(unsigned char* pixels, int h, int w, const char* path, std::vector<MNNRect> &result);
int faceDetectMainR1801(unsigned char* pixels, int h, int w, const char* path, std::vector<MNNRect> &result);

// 输入图像， 图像高与宽，模型路径， 输入算法运行状态开关， 输出人脸或者头肩检测结果, 输出跟踪和手势的识别结果
/* 0 video; 1 nohand; 2 picture; 3 track */
int pv_visual_main(unsigned char* pixels, int h, int w, const char* path,
    std::vector<MNNRect> &faceResult,
    std::vector<MNNRect> &result);

// 新版本号说明：v1表示视觉大版本号，hdrt表示头肩、手势检测、手势识别、跟踪模型版本号，最后的01表示修复的bug或者新增内容，非大更改
// 版本号V1.SDRT01010101.15  // 头肩局部检测(后期修改为320*320，而非320*256)



//#ifdef __cplusplus
//}
//#endif
