//
//  dragRectView.m
//  fftwDemo
//
//  Created by mengchen on 2020/8/11.
//  Copyright © 2020 mengchen. All rights reserved.
//

#import "dragRectView.h"
#include "imgProcess.h"

@interface dragRectView ()

@property (nonatomic, strong) CAShapeLayer * rectShapeLayer;

@property (nonatomic, assign) CGPoint leftTopPoint;

@end

@implementation dragRectView

- (instancetype)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {
        [self.layer addSublayer:self.rectShapeLayer];
    }
    return self;
}

- (void)setPixSize:(CGSize)pixSize{
    _pixSize = pixSize;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    UITouch * touch = [touches anyObject];
    CGPoint currentPoint = [touch locationInView:self];
    self.leftTopPoint = currentPoint;
     [self.layer addSublayer:self.rectShapeLayer];    
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
//    NSLog(@"移动");
    UITouch * touch = [touches anyObject];
    CGPoint currentPoint = [touch locationInView:self];
    CGRect rect = [self followRect:currentPoint];
    UIBezierPath * rectPath = [UIBezierPath bezierPathWithRect:rect];
    
//    CGFloat X = self.layer.frame.size.height - self.leftTopPoint.y;
//    CGFloat Y = self.leftTopPoint.x;
//    CGFloat W = currentPoint.y - self.leftTopPoint.y;
//    CGFloat H = currentPoint.x-self.leftTopPoint.x;
//
//    CGFloat xScale = X/self.layer.frame.size.height;
//    CGFloat yScale = Y/self.layer.frame.size.width;
//    CGFloat wScale = W/self.layer.frame.size.height;
//    CGFloat hScale = H/self.layer.frame.size.width;
//
//    self.rectR = CGRectMake(xScale, yScale, wScale, hScale);
//
//    NSLog(@"x:%f,y:%f,w:%f,h:%f,",xScale,yScale,wScale,hScale);
    
    
    
    self.rectR = rect;
    self.rectShapeLayer.path = rectPath.CGPath;
    
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    
    self.rectR = [self changeRectWithOrientation:self.rectR];
    
    std::vector<accept_data> rectMnn;
    accept_data rect1;
    
    rect1.x = self.rectR.origin.x;
    rect1.y = self.rectR.origin.y;
    rect1.w = self.rectR.size.width;
    rect1.h = self.rectR.size.height;
    rect1.initflag = 1;
    rect1.track_status = 1;
    rect1.hand_status = 1;
    rectMnn.push_back(rect1);
    
    
    acceptData(rectMnn, self.pixSize.height, self.pixSize.width);

    [self.rectShapeLayer removeFromSuperlayer];
}

-(CGRect)followRect:(CGPoint) movePoint{
    CGFloat pathX,pathY,pathW,pathH;
    if (movePoint.y > _leftTopPoint.y) {
        
        if (movePoint.x > _leftTopPoint.x) {
            pathX = _leftTopPoint.x;
            pathY = _leftTopPoint.y;
            pathW = movePoint.x - _leftTopPoint.x;
            pathH = movePoint.y - _leftTopPoint.y;
        } else {
            pathX = movePoint.x;
            pathY = _leftTopPoint.y;
            pathW =  _leftTopPoint.x - movePoint.x;
            pathH = movePoint.y - _leftTopPoint.y;
        }
        
    } else {
        
        if (movePoint.x > _leftTopPoint.x) {
               pathX = _leftTopPoint.x;
               pathY = movePoint.y;
               pathW = movePoint.x - _leftTopPoint.x;
               pathH = _leftTopPoint.y - movePoint.y;
           } else {
               pathX = movePoint.x;
               pathY = movePoint.y;
               pathW =  _leftTopPoint.x - movePoint.x;
               pathH = _leftTopPoint.y - movePoint.y;
           }
    
        
    }
    
    
    return CGRectMake(pathX, pathY, pathW, pathH);;
}

- (UIView*)hitTest:(CGPoint)point withEvent:(UIEvent *)event{
    UIView *hitView = [super hitTest:point withEvent:event];
    if(hitView == self){
        return self;
    }
    return hitView;
}
- (CAShapeLayer *)rectShapeLayer{
    if (!_rectShapeLayer) {
        _rectShapeLayer = [CAShapeLayer layer];
        _rectShapeLayer.fillColor = [UIColor clearColor].CGColor;
        _rectShapeLayer.strokeColor = [UIColor blackColor].CGColor;
        _rectShapeLayer.lineWidth = 2;
    }
    return _rectShapeLayer;
}

- (CGRect)changeRectWithOrientation:(CGRect) rect{
    CGRect newRect = rect;
    CGFloat x,y,w,h,viewH,viewW;
    viewH = self.layer.frame.size.height;
    viewW = self.layer.frame.size.width;
    switch (self.curOrientation) {
            
        case UIDeviceOrientationFaceUp:
            
            NSLog(@"屏幕朝上平躺");
            x = rect.origin.x /viewW;
            y = rect.origin.y / viewH;
            w = rect.size.width / viewW;
            h = rect.size.height / viewH;
            newRect = CGRectMake(x, y, w, h);
            break;
            
        case UIDeviceOrientationFaceDown:
            
            NSLog(@"屏幕朝下平躺");
            x = rect.origin.x /viewW;
            y = rect.origin.y / viewH;
            w = rect.size.width / viewW;
            h = rect.size.height / viewH;
            newRect = CGRectMake(x, y, w, h);
            break;
            
        case UIDeviceOrientationUnknown:
            
            NSLog(@"未知方向");
            x = rect.origin.x /viewW;
            y = rect.origin.y / viewH;
            w = rect.size.width / viewW;
            h = rect.size.height / viewH;
            newRect = CGRectMake(x, y, w, h);
            
            break;
            
        case UIDeviceOrientationLandscapeLeft:{
            NSLog(@"屏幕向左橫置");
           
            
            x = rect.origin.y / viewH;
            y = 1 - (rect.origin.x + rect.size.width) / viewW;
            w = rect.size.height / viewH;
            h = rect.size.width / viewW;
            newRect = CGRectMake(x, y, w, h);
        }
            
            
            
            break;
            
        case UIDeviceOrientationLandscapeRight:{
            NSLog(@"屏幕向右橫置");
            x = 1 - (rect.size.height + rect.origin.y) / viewH;
            y = rect.origin.x / viewW;
            w = rect.size.height / viewH;
            h = rect.size.width / viewW;
            
            newRect = CGRectMake(x, y, w, h);
        }
            
            
            
            
            break;
            
        case UIDeviceOrientationPortrait:{
            NSLog(@"屏幕直立");
            x = rect.origin.x /viewW;
            y = rect.origin.y / viewH;
            w = rect.size.width / viewW;
            h = rect.size.height / viewH;
            newRect = CGRectMake(x, y, w, h);
        }
            
            
            break;
            
        case UIDeviceOrientationPortraitUpsideDown:{
            
            x = 1 - (rect.origin.x + rect.size.width) / viewW;
            y = 1 - (rect.size.height + rect.origin.y) / viewH;
            h = rect.size.height / viewH;
            w = rect.size.width / viewW;
            newRect = CGRectMake(x, y, w, h);
        }
            
            
            
            break;
            
        default:
            
            break;
            
    }
    
    return  newRect;
    
}

@end
