//
//  FaceRectView.m
//  fftwDemo
//
//  Created by mengchen on 2020/8/5.
//  Copyright © 2020 mengchen. All rights reserved.
//

#import "FaceRectView.h"
#define ScreenWidth                         [[UIScreen mainScreen] bounds].size.width
#define ScreenHeight                        [[UIScreen mainScreen] bounds].size.height

@implementation FaceRectModel


@end


@interface FaceRectView ()


@property (nonatomic, strong) UILabel * handStatusLab;

@property (nonatomic, assign) CGSize pixSize;

@end


@implementation FaceRectView

- (instancetype)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {
        [self addSubview:self.handStatusLab];
    }
    return self;
}

- (void)setPixbufferW_H:(CGSize)size{
    self.pixSize = size;
}

-(void)setFacesResult:(NSMutableArray<FaceRectModel *> *)facesResult{
    _facesResult = facesResult;
    [self setNeedsDisplay];
    [self setNeedsLayout];
}

-(void)drawRect:(CGRect)rect{
    //获得处理的上下文
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    //设置线条样式
    CGContextSetLineCap(ctx, kCGLineCapSquare);
    
    
    CGContextSetRGBStrokeColor(ctx, 1.0, 0.0, 0.0, 1.0);
    
    CGContextSetLineWidth(ctx, 1);
    
    float w = self.pixSize.width;
    float h = self.pixSize.height;
    
    
//    if (ScreenWidth>ScreenHeight) {
//        h = self.pixSize.width;
//        w = self.pixSize.height;
//    }
    float kw = self.frame.size.width/h;
    float kh = self.frame.size.height/w;
    CGRect frame = self.frame;
    for(int i = 0; i < _facesResult.count; i++) {
        FaceRectModel *outputModel = _facesResult[i];
        int x = outputModel.x *kw;
        int y = outputModel.y *kh;
        int w = outputModel.w *kw;
        int h = outputModel.h *kh;
        //        int x = outputModel.x;
        //        int y = outputModel.y;
        //        int w = outputModel.w;
        //        int h = outputModel.h;
        
//        NSLog(@"转换前X:%f  转换后X：%d",outputModel.x,x);
//        NSLog(@"转换前Y:%f  转换后Y：%d",outputModel.y,y);
//        NSLog(@"转换前W:%f  转换后W:%d",outputModel.w,w);
//        NSLog(@"转换前H:%f  转换后H:%d",outputModel.h,h);
        
        CGContextStrokeRect(ctx, CGRectMake(x, y, w, h));
        
        
        switch (outputModel.handFlag) {//0 竖八, 1负样本，2剪刀手，3跟随
            case 0:
            {
                self.handStatusLab.text = @"竖八";
            }
                break;
            case 1:
            {
                self.handStatusLab.text = @"负样本";
            }
                break;
            case 2:
            {
                self.handStatusLab.text = @"剪刀手";
            }
                break;
            case 3:
            {
                self.handStatusLab.text = @"跟随";
            }
                break;
                
            default:
                break;
        }
        
        
        
        //        // score
        //        UIFont *font = [UIFont systemFontOfSize:14.0];
        //        NSMutableParagraphStyle *paragraphStyle = [[NSParagraphStyle defaultParagraphStyle] mutableCopy];
        //        [paragraphStyle setAlignment:NSTextAlignmentLeft];
        //        CGRect iRect = CGRectMake(x, y-16, w, 20);
        //        [str drawInRect:iRect withAttributes:@{NSFontAttributeName:font,NSParagraphStyleAttributeName:paragraphStyle,NSForegroundColorAttributeName:color}];
        
    }
}
-(UILabel *)handStatusLab{
    if (!_handStatusLab) {
        _handStatusLab = [[UILabel alloc] initWithFrame:CGRectMake(0, 0, 100, 100)];
    }
    return _handStatusLab;
}
@end
