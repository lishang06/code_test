#ifndef sMatrix_H
#define sMatrix_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include<stdio.h>
#include<stdlib.h>
#include "fftw3.h"
#include <cstring>
using namespace std;


struct sRect{
public:
    float x;
    float y;
    int width;
    int height;
    sRect(void){}
    sRect(int a,int b,int c,int d) {x=a;y=b;width=c;height=d;}
    sRect& operator=(const sRect& A){this->x=A.x;this->y=A.y;this->width=A.width;this->height=A.height; return *this;}
};

struct sSize {
public:
    float w;
    float h;
    sSize(void){}
    sSize(int a,int b) {w=a;h=b;}
    sSize& operator=(const sSize& A){this->w=A.w;this->h=A.h; return *this;}
};
struct sPoint2i{
public:
    int x;
    int y;
    sPoint2i(void){}
    sPoint2i(int a,int b) {x=a;y=b;}
    sPoint2i& operator=(const sPoint2i& A){this->x=A.x;this->y=A.y; return *this;}
};
struct sPoint2f {
public:
    float x;
    float y;
    sPoint2f(void){}
    sPoint2f(float a,float b) {x=a;y=b;}
    sPoint2f& operator=(const sPoint2f& A){this->x=A.x;this->y=A.y; return *this;}
};
struct sPoint3f {
public:
    float x;
    float y;
    float z;
    sPoint3f(void){}
    sPoint3f(int a,int b,int c) {x=a;y=b;z=c;}
    sPoint3f& operator=(const sPoint3f& A){this->x=A.x;this->y=A.y;this->z=A.z; return *this;}
};

template <class T>
class sMatrix
{
public :
    sMatrix()
    {
        data=NULL;rows= cols= 0;
    }

    sMatrix(int w,int h)
    {
        data=NULL;
        if(w>0&&h>0){
            rows=h;
            cols=w;
            data=(T*)malloc(sizeof(T)*w*h);
            memset(data,0,sizeof(T)*w*h);
        }
        else{
            rows= cols= 0;
            printf("error! data size <= 0 !\n");
        }
    }

    sMatrix(int w,int h,T*in)
    {
        data=NULL;
        if(w>0&&h>0){
            rows=h;
            cols=w;
            data = in;
        }
        else{
            rows= cols= 0;
            printf("error! data size <= 0 !\n");
        }
    }

    ~sMatrix( )
    {
    }

    void release( )
    {
        if(data!=NULL){
            free(data);
            data=NULL;
        }
        rows= cols= 0;
    }

    template <class ty>
    sMatrix(const sMatrix<ty>& A)  //copy
    {

        data=NULL;
        if(A.data==NULL||A.cols<=0||A.rows<=0)
        {
            data=NULL;rows= cols= 0;
        }
        else{
            rows=A.rows;cols=A.cols;
            ty*ptr = A.data;
            data=(T*)malloc(sizeof(T)*rows*cols);

            T*data_ptr =  data;
            for(int i =0;i< rows* cols;++i)
            {
                (*data_ptr) =(*ptr) ;ptr++ ;
                data_ptr++;
            }
        }
    }

    T&operator ()(int a,int b)
    {
        if(a<0||b<0||a>rows-1||b>cols-1){
            printf("data out of range\n ");
        }
        return  data[a*cols+b];
    }

    sMatrix<T>& operator=(const sMatrix<T>& A){
        if(A.data==NULL||A.cols<=0||A.rows<=0){
            this->data=NULL; this->rows=0;this->cols=0;
        }
        else{
            this->rows=A.rows;this->cols=A.cols;
            if(this->data!=NULL)
            {
                free(this->data);
                this->data=NULL;
            }
            this->data  = A.data ;
        }
        return *this;
    }
    bool drawRectangle(const sRect&roi,int linewidth){
        if(data==NULL||cols<=0||rows<=0){
            return 0;
        }
        int i,j;

        j=roi.y;
        for(int i=roi.x;i<roi.x+roi.width;++i)
            for(int k = -linewidth/2;k<=linewidth/2;k++)
            {
                int ind = (j+k)*cols+(i);
                if(ind>=0&&ind<rows*cols)
                    data[ind]=0;
            }

        j=roi.y+roi.height;
        for(int i=roi.x;i<roi.x+roi.width;++i)
            for(int k = -linewidth/2;k<=linewidth/2;k++)
            {
                int ind =(j+k)*cols+(i) ;
                if(ind>=0&&ind<rows*cols)
                    data[ind]=0;
            }


        i=roi.x;
        for(int j=roi.y;j<roi.y+roi.height;++j)
            for(int k = -linewidth/2;k<=linewidth/2;k++)
            {
                int ind =(j)*cols+i+k ;
                if(ind>=0&&ind<rows*cols)
                    data[ind]=0;
            }

        i=roi.x+roi.width;
        for(int j=roi.y;j<roi.y+roi.height;++j)
            for(int k = -linewidth/2;k<=linewidth/2;k++)
            {
                int ind =(j)*cols+i+k;
                if(ind>=0&&ind<rows*cols)
                    data[ind]=0;
            }
        return 1;
    }


    sMatrix<T> resize( int w1,int h1){
        sMatrix<T> dst (w1,h1);
        if( data==NULL|| cols<=0|| rows<=0||w1<=0||h1<=0){
            return dst;
            printf("error! data size <= 0 !\n");
        }
        T* pSrc = data;
        int w0 = cols;
        int h0 = rows;

        T* p0, *p1 = dst.data;

        float fw = float(w0-1) / (w1-1);
        float fh = float(h0-1) / (h1-1);
        int pitch0=w0,pitch1 = w1;
        float x0, y0;
        int y1, y2, x1, x2;
        float fx1, fx2, fy1, fy2;


        int* arr_x2 = (int *)malloc(w1 *sizeof(int));
        int* arr_x1 = (int *)malloc(w1 *sizeof(int));
        float* arr_fx1  =(float *)malloc(w1 *sizeof(float));


        for(int x=0; x<w1; x++)
        {
            x0 = x*fw;
            arr_x1[x] = int(x0);
            arr_x2[x] = int(x0+0.5f);
            arr_fx1[x] = x0 - arr_x1[x];
            //TRACE(L"x=%6d; x0=%6.3f; x1=%6d; x2=%6d; fx1=%6.3f;\n", x, x0, arr_x1[x], arr_x2[x], arr_fx1[x]);
        }
        for(int y=0; y<h1; y++)
        {
            y0 = y*fh;
            y1 = int(y0);
            y2 = int(y0+0.5f);
            fy1 = y0-y1;
            fy2 = 1.0f - fy1;
            //TRACE(L"y=%6d; y0=%6.3f; y1=%6d; y2=%6d; fy1=%6.3f;\n", y, y0, y1, y2, fy1);
            for(int x=0; x<w1; x++)
            {
                x1 = arr_x1[x];
                x2 = arr_x2[x];
                fx1 = arr_fx1[x];
                fx2 = 1.0f-fx1;

                float s1 = fx2*fy2;
                float s2 = fx1*fy2;
                float s3 = fx1*fy1;
                float s4 = fx2*fy1;
                //TRACE(L"s1=%6.3f; s2=%6.3f; s3=%6.3f; s4=%6.3f; sum=%6.3f\n", s1,s2,s3,s4, s1+s2+s3+s4);
                T* p11 = pSrc + pitch0*y1 + x1;
                T* p12 = pSrc + pitch0*y1 + x2;
                T* p21 = pSrc + pitch0*y2 + x1;
                T* p22 = pSrc + pitch0*y2 + x2;

                *p1 = T((*p11)*s1 + (*p12)*s2 + (*p21)*s4 + (*p22)*s3);
                p1++;
                //            *p1 = T((*p11)*s1 + (*p12)*s2 + (*p21)*s4 + (*p22)*s3);    p1++;    p11++; p12++; p21++; p22++;
                //            *p1 = T((*p11)*s1 + (*p12)*s2 + (*p21)*s4 + (*p22)*s3);    p1++;    p11++; p12++; p21++; p22++;
                //            *p1 = T((*p11)*s1 + (*p12)*s2 + (*p21)*s4 + (*p22)*s3);    p1++;

            }
            //        p1 = res + y*pitch1;
        }

        delete  []arr_x2;
        delete  []arr_x1;
        delete  []arr_fx1;
        //        free(arr_x1);
        //        free(arr_x2);
        //        free(arr_fx1);


        return dst;
    }




    sMatrix<T> copy(){
        if( data==NULL|| cols<=0|| rows<=0 ){
            printf("error! data size <= 0 !\n");
            sMatrix<T> dst;
            return dst;
        }
        sMatrix<T> dst ( cols, rows);
        T*ptr =  data;

        int s =rows* cols;
        for(int i =0;i< s;++i)
        {
            data[i] =(*ptr) ;ptr++ ;
        }
        return  dst;
    }


    void save2txt(const char*name){
        ofstream outfile;
        outfile.open(name);
        if(!outfile){
            cout<<"open txt error"<<endl;
            return;
        }

        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j)
                outfile<< (int)data[i* cols+j]<<"   ";
            outfile<<"\n";
        }
        outfile.close();
    }

    sMatrix<T>cutRect( const sRect &roi  )//FIBITMAP *image)
    {
        sMatrix<T> res;
        if(roi.width <= 0 ||roi.height<= 0 || roi.x <  0|| roi.y<  0     )
        {
            printf("cut roi size Error  : size<=0\n ");
            return res ;
        }
        if(roi.y+roi.height >rows  ||roi.x+roi.width >cols       )
        {
            printf("cut roi size Error  : size too large\n ");
            return res ;
        }
        if(data==NULL)return res;
        float*data_temp=
                new float [roi.width*roi.height ] ;


        int count=0;
        for(int i=roi.y;i<roi.y+roi.height;++i){
            for(int j=roi.x;j<roi.x+roi.width;++j){
                data_temp[count++] =  data[i* cols+j];
            }
        }
        res.data = data_temp;
        res.cols =roi.width;
        res.rows = roi.height;
        return res;
    }


public :
    int rows,cols;
    T*data;
    int depth;
};









class sFftwf{

public:
    sFftwf(void){
        fft_input=NULL;
        fft_output=NULL;
        tmp=NULL;
        ifft_input=NULL;
        ifft_output=NULL;
    }
    sFftwf(int width,int height,int depth) {
        init( width, height,depth);
    }
    void changeSize(int width,int height,int depth){
        if(len_patch[0]!=width||len_patch[1]!=height){
            release();
            init( width, height, depth);
        }
    }
    void release(){
        if(fft_input!=NULL){
            fftwf_free(fft_input);
            fft_input=NULL;
            fftwf_free(fft_output);
            fft_output=NULL;
            fftwf_free(ifft_input);
            ifft_input=NULL;
            fftwf_free(ifft_output);
            ifft_output=NULL;
            fftwf_free(tmp);
            tmp=NULL;
        }
    }

private:
    void init(int width,int height,int d) {
        len_patch[0] = width;
        len_patch[1] = height;
        len_patch[2] = d;

        int len = width*height;
        fft_input  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * len);
        fft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * len);
        fft2_plan = fftwf_plan_dft_2d(height,width, fft_input, fft_output, false ? 1 : -1,  FFTW_ESTIMATE);

        tmp  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft_input   = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft2_plan =fftwf_plan_dft_2d(height,width, ifft_input, ifft_output, true ? 1 : -1,  FFTW_ESTIMATE);
    }
public:
    fftwf_plan fft2_plan,ifft2_plan;
    fftwf_complex * fft_input;
    fftwf_complex * fft_output;
    fftwf_complex * tmp;
    fftwf_complex * ifft_input ;
    fftwf_complex * ifft_output;

    int len_patch[3];
};


class sFftwf1D{

public:
    sFftwf1D(void){
        fft_input=NULL;
        fft_output=NULL;
        tmp=NULL;
        ifft_input=NULL;
        ifft_output=NULL;
    }
    sFftwf1D(int length , int depth) {
        init( length,depth);
    }
    void changeSize(int length,int depth){
        if(len_patch[0]!=length||len_patch[1]!=depth){
            release();
            init( length,depth);
        }
    }
    void release(){
        if(fft_input!=NULL){
            fftwf_free(fft_input);
            fft_input=NULL;
            fftwf_free(fft_output);
            fft_output=NULL;
            fftwf_free(ifft_input);
            ifft_input=NULL;
            fftwf_free(ifft_output);
            ifft_output=NULL;
            fftwf_free(tmp);
            tmp=NULL;
        }
    }

private:
    void init(int length,int d) {
        len_patch[0] = length;
        len_patch[1] = d;

        int len = length;
        fft_input  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * len);
        fft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) * len);
        fft1_plan = fftwf_plan_dft_1d(len, fft_input, fft_output, false ? 1 : -1,  FFTW_ESTIMATE);



        tmp  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft_input   = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft_output  = (fftwf_complex*) fftwf_malloc(sizeof (fftwf_complex) *len);
        ifft1_plan =fftwf_plan_dft_1d(len, ifft_input, ifft_output, true ? 1 : -1,  FFTW_ESTIMATE);

    }
public:
    fftwf_plan fft1_plan,ifft1_plan;
    fftwf_complex * fft_input;
    fftwf_complex * fft_output;
    fftwf_complex * tmp;
    fftwf_complex * ifft_input ;
    fftwf_complex * ifft_output;

    int len_patch[2];
};







typedef sMatrix<float>  sMatf;
typedef sMatrix<int>  sMati;
typedef sMatrix<double>  sMatd;
typedef sMatrix<unsigned char>  sMatc;
typedef sMatrix<fftwf_complex>  sMatcf;




#endif // sMatrix_H
