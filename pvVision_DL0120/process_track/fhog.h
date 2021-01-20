#ifndef FHOGDSST_H
#define FHOGDSST_H

#include <cstdlib>
#include <cmath>
#include <cstring>
#include "sse.hpp"
//#include "sMatrix.h"

/**
    Inputs:
        float* I        - a gray or color image matrix with shape = channel x width x height
        int *h, *w, *d  - return the size of the returned hog features
        int binSize     -[8] spatial bin size
        int nOrients    -[9] number of orientation bins
        float clip      -[.2] value at which to clip histogram bins
        bool crop       -[false] if true crop boundaries

    Return:
        float* H        - computed hog features with shape: (nOrients*3+5) x (w/binSize) x (h/binSize), if not crop

    Author:
        Sophia
    Date:
        2015-01-15
**/

void fhog(float* I,float* H,int height,int width,int channel,int binSize = 4,int nOrients = 7,float clip=0.2f,bool crop = false);
void fhog( float *M, float *O, float *H, int h, int w, int binSize,int nOrients, int softBin, float clip );
void getTrackFhog(float* input, int h , int w, float*  out, int binSize);


void scaleFhog(float* I,float* H,int height,int width,int channel,int binSize = 4,int nOrients = 7,float clip=0.2f,bool crop = false);
void scaleFhog( float *M, float *O, float *H, int h, int w, int binSize,int nOrients, int softBin, float clip );
void getScaleTrackFhog(float* input, int h , int w, float*  out, int binSize);

//void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );
// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc( size_t num, size_t size ) { return calloc(num,size); }
inline void* wrMalloc( size_t size ) { return malloc(size); }
inline void wrFree( void * ptr ) { free(ptr); }

#endif
