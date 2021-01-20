#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

#define BICUBIC_KERNEL 4
static const int TFORM_SIZE = 6;

typedef enum SAMPLING_TYPE
{
    LINEAR, 
    BICUBIC 
} SAMPLING_TYPE;

typedef enum PADDING_TYPE
{
    ZERO_PADDING,           
    NEAREST_PADDING,
} PADDING_TYPE;


const float meanshape[10] = 
{
    38.2946f, 51.6963f,
    73.5318f, 51.5014f,
    56.0252f, 71.7366f,
    41.5493f, 92.3655f,
    70.7299f, 92.2041f
};

void near_sampling( const uint8_t *image_data, int image_width, int image_height, int image_channels, int x, int y, uint8_t *pixel );
void sampling( const uint8_t *image_data, int image_width, int image_height, int image_channels, double scale, double x, double y, uint8_t *pixel,
    std::vector<double> &weights_x, std::vector<double> &weights_y, std::vector<int> &indices_x, std::vector<int> &indices_y, SAMPLING_TYPE LINEAR,
    PADDING_TYPE ZERO_PADDING );

bool spatial_transform( const uint8_t *image_data, int image_width, int image_height, int image_channels, uint8_t *crop_data, int crop_width, int crop_height,
    const double *transformation, int pad_top , int pad_bottom , int pad_left , int pad_right , SAMPLING_TYPE LINEAR, 
    PADDING_TYPE ZERO_PADDING, int N );

bool caculate_rectified_points( const float *points, int points_num, const double *transformation, int pad_top, int pad_left, float *final_points );

bool transformation_maker(int crop_width, int crop_height,const float *points, int points_num, const float *mean_shape, 
    int mean_shape_width, int mean_shape_height,double *transformation, int N );

bool face_crop_core_executor( const uint8_t *image_data, int image_width, int image_height, int image_channels, uint8_t *crop_data, int crop_width, int crop_height,
    const float *points, int points_num, const float *mean_shape, int mean_shape_width, int mean_shape_height, int pad_top, int pad_bottom, int pad_left, int pad_right,
    float *final_points, SAMPLING_TYPE type, PADDING_TYPE ptype );

bool face_crop_core(const uint8_t* image_data, int image_width, int image_height, int image_channels, uint8_t* crop_data, int crop_width, int crop_height, 
     const float* points, int points_num, const float* mean_shape, int mean_shape_width, int mean_shape_height, int pad_top, int pad_bottom, int pad_left, 
     int pad_right, float* final_points, SAMPLING_TYPE type);
