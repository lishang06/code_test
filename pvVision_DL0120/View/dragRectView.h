//
//  dragRectView.h
//  fftwDemo
//
//  Created by mengchen on 2020/8/11.
//  Copyright © 2020 mengchen. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface dragRectView : UIView
@property (nonatomic, assign) CGRect  rectR;
@property (nonatomic, assign) UIDeviceOrientation curOrientation;
@property (nonatomic, assign) CGSize  pixSize;
@end

NS_ASSUME_NONNULL_END
