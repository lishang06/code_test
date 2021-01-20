//
//  FaceRectView.h
//  fftwDemo
//
//  Created by mengchen on 2020/8/5.
//  Copyright © 2020 mengchen. All rights reserved.
//

#import <UIKit/UIKit.h>



NS_ASSUME_NONNULL_BEGIN

@interface FaceRectModel : NSObject
@property (nonatomic, assign) CGFloat x;
@property (nonatomic, assign) CGFloat y;
@property (nonatomic, assign) CGFloat w;
@property (nonatomic, assign) CGFloat h;

@property (nonatomic, assign) NSInteger handFlag; // 手势识别的类型：0大拇指, 1负样本，2剪刀手，3合影，4五指张开

@end

@interface FaceRectView : UIView

@property (nonatomic, strong) NSMutableArray<FaceRectModel*> * facesResult;

/// 设置视频宽高
/// @param size  宽 高
- (void)setPixbufferW_H:(CGSize)size;


@end

NS_ASSUME_NONNULL_END
