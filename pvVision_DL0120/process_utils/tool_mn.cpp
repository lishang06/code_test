#include "tool_mn.h"

using namespace cv;
using namespace std;

/*
* 手机摄像头图像转换
*/
void rotationImg(cv::Mat& src, cv::Mat& dst, int angle, int orientation)
{
    /* 前置摄像头*/
    if( orientation==1 )
    {
        if( angle==0 )                   /* 右 无旋转　*/
        {
            dst = src;
        }
        else if( angle==180 )            /* 左 旋转180　*/
        {
            flip(src,dst,-1);
        }
        else if( angle==90 )             /* 上　旋转-90　*/
        {
            Mat temp;
            transpose(src, temp);
            flip(temp,dst,1);
        }
        else if( angle==-90 )            /* 上　旋转90　*/
        {
            Mat temp;
            transpose(src, temp);
            flip(temp,dst,0);
        }
    }

    /*　后置摄像头　*/
    if( orientation==0 )
    {
        if( angle==180 )                   /* 左 无旋转　*/
        {
            dst = src;
        }
        else if( angle==0 )            /* 右　旋转180　*/
        {
            flip(src,dst,-1);
        }
        else if( angle==90 )             /* 上　旋转-90　*/
        {
            Mat temp;
            transpose(src, temp);
            flip(temp,dst,1);
        }
        else if( angle==-90 )            /* 上　旋转90　*/
        {
            Mat temp;
            transpose(src, temp);
            flip(temp,dst,0);
        }
    }
}

/*
* 发送给app的坐标转换，统一发送竖屏模式下的坐标
*/
void rotationLocation(int* location, int w, int h, int diff, int angle, int orientation, int padding ,int img_w, int img_h)
{
    /* 前置摄像头*/
    if( orientation==1 )
    {
        if( angle==0 || angle==180 )       /* 左／右 坐标转换　*/
        {
            /*  坐标旋转 */
            if( angle==180 )
            {
                location[0] = w-location[0]-location[2];
                //location[1] = h-location[1]-location[3];
                location[2] = location[2];
                //location[3] = location[3];
                if((img_w-img_h)<1){
                    location[1] = (h-location[1]-location[3]);
                    location[3] = location[3]*3;
                }
                else if(img_w*1.0/img_h>1.5){
                    location[1] = img_h-location[1]-location[3];
                    location[3] = location[3];
                }
                else{
                    location[1] = h-location[1]-location[3];
                    location[3] = location[3];
                }
            }
            else{
                if((img_w-img_h)<1){
                    location[1] = location[1];
                    location[3] = location[3];
                }
                else if(img_w*1.0/img_h>1.5){
                    location[1] = location[1];
                    location[3] = location[3];
                }
                else{
                    location[1] = location[1];
                    location[3] = location[3];
                }
            }
            
            int xtmp = location[0];
            location[0] = location[1];
            location[1] = w - (xtmp + location[2]);

            int wtmp = location[2];
            location[2] = location[3];
            location[3] = wtmp;
        }
        else if( angle==90 || angle==-90 )  /* 上下 不变　*/
        {
            if( angle==90 )
            {
                if((img_w-img_h)<1){
                    location[0] = w - diff - location[0] - location[2];
                    location[1] = h - location[1] - location[3];
                    location[2] = location[2];
                }
                else if( img_w*1.0/img_h>1.5 ){
                    location[0] = w - diff - location[0] - location[2];
                    location[1] = h - location[1] - location[3];
                    location[2] = location[2];
                }
                else{
                    location[0] = w - diff - location[0] - location[2];
                    location[1] = h - location[1] - location[3];
                }
                
            }
            else{
                if((img_w-img_h)<1){
                    location[0] = location[0];
                    location[2] = location[2];
                }
                if( img_w*1.0/img_h>1.5 ){
                    location[0] = location[0];
                    location[2] = location[2];
                }
            }
        }
    }

    /* 后置摄像头*/
    if( orientation==0 )
    {
        if( angle==0 || angle==180 )   /* 左／右 坐标转换　*/
        {
            /*  坐标旋转 */
            if( angle==0 )
            {
                location[0] = w-location[0]-location[2];
                location[1] = location[1];
                location[2] = location[2];
                location[3] = location[3];
                if(img_w*1.0/img_h>1.5){
                    location[1] = location[1];
                    location[3] = location[3];
                }
                if((img_w-img_h)<1){
                    location[1] = location[1];
                    location[3] = location[3];
                }
            }else
            {
                if(img_w*1.0/img_h>1.5){
                    location[1] = h*3.0/4.0-(location[1]+location[3]);
                    location[3] = location[3];
                }
                else if((img_w-img_h)<1){
                    location[1] = (h-location[1]-location[3]);
                    location[3] = location[3];
                }
                else{
                    location[1] = h-location[1]-location[3];
                }
                
            }
            
            int xtmp = location[0];
            location[0] = location[1];
            location[1] = w - (xtmp + location[2]);

            int wtmp = location[2];
            location[2] = location[3];
            location[3] = wtmp;
        }
        else if( angle==90 || angle==-90 )  /* 上下 不变　*/
        {
            if( angle==-90 )
            {
                if((img_w-img_h)<1){
                    location[0] = w - diff - (location[0] + location[2]);
                    location[2] = location[2];
                }
                else if(img_w*1.0/img_h>1.5){
                    location[0] = w - diff - (location[0] + location[2]);
                    location[2] = location[2];
                }
                else {
                    location[0] = w - diff - location[0] - location[2];
                }
            }
            else
            {
                location[1] = h - location[1] - location[3];
                if(img_w*1.0/img_h>1.5){
                    location[0] = location[0];
                    location[2] = location[2];
                }
                if((img_w-img_h)<1){
                    location[0] = location[0];
                    location[2] = location[2];
                }
                
            }
        }
    }
}
void rotationLocation_copy(int* location, int w, int h, int diff, int angle, int orientation, int padding)
{
    /* 前置摄像头*/
    if( orientation==1 )
    {
        if( angle==0 || angle==180 )       /* 左／右 坐标转换　*/
        {
            /*  坐标旋转 */
            if( angle==180 )
            {
                location[0] = w-location[0]-location[2];
                location[1] = h-location[1]-location[3];
                location[2] = location[2];
                location[3] = location[3];
            }
            
            int xtmp = location[0];
            location[0] = location[1];
            location[1] = w - (xtmp + location[2]);

            int wtmp = location[2];
            location[2] = location[3];
            location[3] = wtmp;
        }
        else if( angle==90 || angle==-90 )  /* 上下 不变　*/
        {
            if( angle==90 )
            {
                location[0] = w - diff - location[0] - location[2];
                location[1] = h - location[1] - location[3];
                //printf(" location w h x y w h is %d, %d, %d, %d, %d, %d\n",w, h, location[0],location[1],location[2],location[3] );
            }
        }
    }

    /* 后置摄像头*/
    if( orientation==0 )
    {
        if( angle==0 || angle==180 )   /* 左／右 坐标转换　*/
        {
            /*  坐标旋转 */
            if( angle==0 )
            {
                location[0] = w-location[0]-location[2];
                location[1] = location[1];
                location[2] = location[2];
                location[3] = location[3];
            }else
            {
                location[1] = h-location[1]-location[3];
            }
            
            int xtmp = location[0];
            location[0] = location[1];
            location[1] = w - (xtmp + location[2]);

            int wtmp = location[2];
            location[2] = location[3];
            location[3] = wtmp;
        }
        else if( angle==90 || angle==-90 )  /* 上下 不变　*/
        {
            if( angle==-90 )
            {
                location[0] = w - diff - location[0] - location[2];
            }
            else
            {
                location[1] = h - location[1] - location[3];
            }
        }
    }
}
/*
* 图像pading操作,旋转图像后再进行padding操作
* 将图像转换为1:1
*/
void imgPpading(cv::Mat& src, cv::Mat& dst, int angle, int orientation)
{
    int top, bottom, left, right;
    dst = src;

    /* 4:3或者16:9 */
    if( src.cols > src.rows )
    {
        top = 0;
        bottom = src.cols - src.rows;
        left = 0;
        right = 0;
        if( angle==90 || angle==-90 )         // 图像添加pading操作
        {
            copyMakeBorder(src,dst,top,bottom,left,right,cv::BORDER_CONSTANT,Scalar(0, 0, 0));
        }
        else if( angle==180 || angle==0 )　　　// 不操作
        {
            // sudo
            if(src.rows<480){
                copyMakeBorder(src,dst,top,480 - src.rows,left,right,cv::BORDER_CONSTANT,Scalar(0, 0, 0));
            }
        }
    }
    else if( src.cols < src.rows )
    {
        top = 0;
        bottom = 0;
        left = 0;
        right = src.rows - src.cols;
        if( angle==90 || angle==-90 )         // 图像添加pading操作
        {
            copyMakeBorder(src,dst,top,bottom,left,right,cv::BORDER_CONSTANT,Scalar(0, 0, 0));
        }
        else if( angle==180 || angle==0 )　　　// 不操作
        {
            // sudo
        }
    }
}

/*
 * initTrack_rotation
 */
int initTrack_rotation(float* init_location, int src_w, int src_h, int dst_w, int dst_h)
{
    /* 状态1 (1,0) right */
    printf("%f, %f, %f, %f, %d, %d, %d, %d\n",init_location[0],init_location[1],init_location[2],init_location[3], dst_w, dst_h, src_w, src_h);
    init_location[0] = (int)(init_location[0]*dst_w/src_w);
    init_location[1] = (int)(init_location[1]*dst_h/src_h);
    init_location[2] = (int)(init_location[2]*dst_w/src_w);
    init_location[3] = (int)(init_location[3]*dst_h/src_h);

//    /* 状态2 (1,180) left */
//    init_location[0] = (int)(init_location[0]*dst_w/src_w);
//    init_location[1] = (int)(init_location[1]*dst_h/src_h);
//    init_location[2] = (int)(init_location[2]*dst_w/src_w);
//    init_location[3] = (int)(init_location[3]*dst_h/src_h);
//
//    /* 状态3 (1,-90) up */
//    init_location[0] = (int)(init_location[0]*dst_w/src_w);
//    init_location[1] = (int)(init_location[1]*dst_h/src_h);
//    init_location[2] = (int)(init_location[2]*dst_w/src_w);
//    init_location[3] = (int)(init_location[3]*dst_h/src_h);
//
//    /* 状态4 (1,90) bottom */
//    init_location[0] = (int)(init_location[0]*dst_w/src_w);
//    init_location[1] = (int)(init_location[1]*dst_h/src_h);
//    init_location[2] = (int)(init_location[2]*dst_w/src_w);
//    init_location[3] = (int)(init_location[3]*dst_h/src_h);
    
    return 0;
}


/*
 * 这里的原始坐标是发送给app的坐标（原始图像坐标上的坐标）
 * 新的坐标是当前旋转角对应的新坐标(640*480图像上的坐标)
 * src w h为app发送的w，h
 * 暂时假设输入也是640*480
 */
void currentAngleLocation(float* location, float* result, int angle, int orint, int src_w, int src_h, int dst_w, int dst_h, int img_w, int img_h )
{
    printf("currentAngleLocation src_w src_h dst_w dst_h img_w img_h is %d, %d, %d, %d, %d, %d\n",src_w, src_h, dst_w, dst_h, img_w, img_h );
    printf("currentAngleLocation angle is %d\n",angle);
    /* 前置摄像头 */
    if( orint==1 )
    {
        if( angle==-90 ) // ok
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            result[0] = result[0] * img_h;
            result[1] = result[1] * img_w;
            result[2] = result[2] * img_h;
            result[3] = result[3] * img_w;
            
            result[0] = result[0] * dst_w/src_w;
            result[1] = result[1] * dst_h/src_h;
            result[2] = result[2] * dst_w/src_w;
            result[3] = result[3] * dst_h/src_h;
            
            if((img_w-img_h)<1){
                result[0] = result[0];
                result[2] = result[2];
            }else if( img_w*1.0/img_h>1.5 )
            {
                result[0] = location[0];
                result[2] = location[2];
            }
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
        }
        else if( angle==0 )  // ok
        {
            printf("location is %f, %f, %f, %f\n",location[0],location[1],location[2],location[3]);
            /* 坐标旋转 */
            result[0] = src_w - location[1] / 3.0 * 4.0 - location[3] / 3 * 4.0;
            result[1] = location[0] ;
            result[2] = location[3] / 3.0 * 4.0;
            result[3] = location[2] ;
            
            if(img_w*1.0/img_h>1.5){
                result[1] = location[0];
                result[3] = location[2];
            }
            if((img_w-img_h)<1){
                result[1] = location[0]*3.0/4.0;
                result[3] = location[2]*3.0/4.0;
                result[0] = dst_w - location[1] - location[3];
                result[2] = location[3];
                
            }
            printf("location1 is %f, %f, %f, %f\n",result[0],result[1],result[2],result[3]);
            
        }else if( angle==180 )
        {
            result[0] = location[1] / 3.0 * 4.0;
            result[1] = src_h - location[0] - location[2] ;
            result[2] = location[3] / 3.0 * 4.0 ;
            result[3] = location[2] ;
            if(img_w*1.0/img_h>1.5){
                //result[1] = result[1] *3.0/4.0;
                //result[3] = result[3] *3.0/4.0;
                result[1] = img_h - location[0] - location[2];
                result[3] = location[2];
            }
            if((img_w-img_h)<1){
                result[0] = location[1];
                result[1] = dst_h - (location[0] - location[2])*3.0/4.0;
                result[2] = location[3];
                result[3] = location[2]*3.0/4.0;
            }
        }
        else if( angle==90 ) // ok 但是转过去后，app上高太高
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            result[0] = result[0] * img_h;
            result[1] = result[1] * img_w;
            result[2] = result[2] * img_h;
            result[3] = result[3] * img_w;
            
            result[0] = result[0] * dst_w/src_w;
            result[1] = result[1] * dst_h/src_h;
            result[2] = result[2] * dst_w/src_w;
            result[3] = result[3] * dst_h/src_h;
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 颠倒镜像 */
            if((img_w-img_h)<1){
                result[0] = dst_w - (result[0] + result[2]);
                result[2] = result[2];
            }
            else if( img_w*1.0/img_h>1.5 ){
                result[0] = img_h - location[0] - location[2];
                result[2] = location[2];
            }else{
                result[0] = img_h - result[0] - result[2];
                result[2] = result[2];
                result[3] = result[3];
            }
        }
    }
    else  /* 后置摄像头 */
    {
        if( angle==-90 )  // ok
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            result[0] = result[0] * img_h;
            result[1] = result[1] * img_w;
            result[2] = result[2] * img_h;
            result[3] = result[3] * img_w;
            
            result[0] = result[0] * dst_w/src_w;
            result[1] = result[1] * dst_h/src_h;
            result[2] = result[2] * dst_w/src_w;
            result[3] = result[3] * dst_h/src_h;
            
            if((img_w-img_h)<1){
                result[0] = dst_w - (result[0] + result[2]);
                result[2] = result[2];
            }else if( img_w*1.0/img_h>1.5 )
            {
                result[0] = img_h - (location[0]+location[2])*3.0/4.0;
                result[2] = location[2]*3.0/4.0;
            }
            else{
                result[0] = (dst_w-location[0]-location[2])*3.0/4.0;
                result[2] = location[2]/3.0*4.0;
            }
                
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
        }
        else if( angle == 0)  // ok
        {
            
            result[0] = location[1] / 3.0 * 4.0;
            result[1] = location[0];
            result[2] = location[3] / 3.0 * 4.0 ;
            result[3] = location[2];
            if(img_w*1.0/img_h>1.5){
                result[1] = location[0];
                result[3] = location[2];
            }
            if((img_w-img_h)<1){
                result[0] = location[1];
                result[1] = location[0]*3.0/4.0;
                result[2] = location[3];
                result[3] = location[2]*3.0/4.0;
                
            }
        }
        else if( angle == 180 )
        {
            printf("location11 is %f, %f, %f, %f\n",location[0], location[1], location[2], location[3]);
          
            result[0] = src_w - location[1] / 3.0 * 4.0 - location[3] / 3.0 * 4.0;
            result[1] = src_h - location[0] - location[2];
            result[2] = location[3] / 3.0 * 4.0;
            result[3] = location[2] ;
            if(img_w*1.0/img_h>1.5){
                result[1] = img_h-location[0]-location[2];
                result[3] = location[2];
            }
            if((img_w-img_h)<1){
                result[0] = dst_w - location[1] - location[3];
                result[1] = dst_h - (location[0] - location[2])*3.0/4.0;
                result[2] = location[3];
                result[3] = location[2]*3.0/4.0;
            }
        }
        else if( angle==90 )  //ok
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            result[0] = result[0] * img_h;
            result[1] = result[1] * img_w;
            result[2] = result[2] * img_h;
            result[3] = result[3] * img_w;
            
            result[0] = result[0] * dst_w/src_w;
            result[1] = result[1] * dst_h/src_h;
            result[2] = result[2] * dst_w/src_w;
            result[3] = result[3] * dst_h/src_h;
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 颠倒镜像 */
            result[1] = dst_h - result[1] - result[3] ;
            if((img_w-img_h)<1){
                result[0]=result[0];
                result[2]=result[2];
            }else if( img_w*1.0/img_h>1.5 )
            {
                result[0] = location[0]*3.0/4.0;
                result[2] = location[2]*3.0/4.0;
            }
        }
    }
}
void currentAngleLocation_copy(float* location, float* result, int angle, int orint, int src_w, int src_h, int dst_w, int dst_h, int img_w, int img_h )
{
    printf("currentAngleLocation src_w src_h dst_w dst_h img_w img_h is %d, %d, %d, %d, %d, %d\n",src_w, src_h, dst_w, dst_h, img_w, img_h );
    printf("currentAngleLocation angle is %d\n",angle);
    /* 前置摄像头 */
    if( orint==1 )
    {
        if( angle==-90 ) // ok
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 扩充padding后的坐标 */
            //result[0] = result[0] * 480 / 640;
            //result[1] = result[1];
            //result[2] = result[2] * 480 / 640;
            //result[3] = result[3];
            //printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* vga上的像素坐标 */
            result[0] = result[0] * dst_w / 4.0 * 3.0;
            result[1] = result[1] * dst_h;
            result[2] = result[2] * dst_w / 4.0 * 3.0;
            result[3] = result[3] * dst_h;
            
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
        }
        else if( angle==0 )  // ok
        {
            printf("location is %f, %f, %f, %f\n",location[0],location[1],location[2],location[3]);
            /* 坐标旋转 */
            result[0] = src_w - location[1] / 3.0 * 4.0 - location[3] / 3 * 4.0;
            result[1] = location[0] ;
            result[2] = location[3] / 3.0 * 4.0;
            result[3] = location[2] ;
            printf("location is %f, %f, %f, %f\n",src_w - location[1] - location[3],location[0],location[2],location[3]);
            printf("location1 is %f, %f, %f, %f\n",result[0],result[1],result[2],result[3]);
            
            /* 扩充padding后的归一化坐标 */
            // todo,横屏模式下不进行扩充,直接输出的就是vga上面的坐标
        }else if( angle==180 )
        {
            result[0] = location[1] / 3.0 * 4.0;
            result[1] = src_h - location[0] - location[2] ;
            result[2] = location[3] / 3.0 * 4.0 ;
            result[3] = location[2] ;
        }
        else if( angle==90 ) // ok 但是转过去后，app上高太高
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 扩充padding后的坐标 */
            //result[0] = result[0] * 480 / 640;
            //result[1] = result[1];
            //result[2] = result[2] * 480 / 640;
            //result[3] = result[3];
            //printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* vga上的像素坐标 */
            result[0] = result[0] * dst_w;
            result[1] = result[1] * dst_h;
            result[2] = result[2] * dst_w;
            result[3] = result[3] * dst_h;
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 颠倒镜像 */
            result[0] = dst_h - result[0]/4.0*3.0 - result[2]/4.0*3.0;
            //result[0] = result[0] / 4.0 * 3.0;
            result[1] = dst_h - result[1] - result[3];
            result[2] = result[2] / 4.0 * 3.0;
            result[3] = result[3];
        }
    }
    else  /* 后置摄像头 */
    {
        if( angle==-90 )  // ok
        {
            
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* vga上的像素坐标 */
            result[0] = result[0] * dst_w;
            result[1] = result[1] * dst_h;
            result[2] = result[2] * dst_w;
            result[3] = result[3] * dst_h;
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);

              
            result[0] = (dst_w - result[0] - result[2]) / 4.0 * 3.0;
            result[2] = result[2] / 4.0 * 3.0;
        }
        else if( angle == 0)  // ok
        {
            //result[0] = src_w - location[1] - location[3];
            result[0] = location[1] / 3.0 * 4.0;
            result[1] = location[0] ;
            //result[2] = location[2] ;
            //result[3] = location[3] / 3.0 * 4.0;
            result[2] = location[3] / 3.0 * 4.0 ;
            result[3] = location[2] ;
        }
        else if( angle == 180 )
        {
            printf("location11 is %f, %f, %f, %f\n",location[0], location[1], location[2], location[3]);
          
            result[0] = src_w - location[1] / 3.0 * 4.0 - location[3] / 3.0 * 4.0;
            result[1] = src_h - location[0] - location[2];
            result[2] = location[3] / 3.0 * 4.0;
            result[3] = location[2] ;
        }
        else if( angle==90 )  //ok
        {
            /* 竖屏归一化后的坐标 */
            result[0] = location[0] / src_w;
            result[1] = location[1] / src_h;
            result[2] = location[2] / src_w;
            result[3] = location[3] / src_h;
            printf("result1 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* vga上的像素坐标 */
            result[0] = result[0] * dst_w;
            result[1] = result[1] * dst_h;
            result[2] = result[2] * dst_w;
            result[3] = result[3] * dst_h;
            printf("result2 is %f, %f, %f, %f\n",result[0], result[1], result[2], result[3]);
            
            /* 颠倒镜像 */
            result[0] = result[0] / 4.0 * 3.0;
            result[1] = dst_h - result[1] - result[3] ;
            result[2] = result[2] / 4.0 * 3.0;
            result[3] = result[3];
        }
    }
}
