# opencv-learning

## 简介

本仓库存放个人学习opencv时写的代码。

编译使用的ide为Visual Studio 2022，使用C++

## 目前为止实现的功能

- RGB_Histgram，调用本机摄像头，实时绘制RGB直方图
- Threshold_Trackbar，创建一个简易滑条实现可变阈值的图像二值化
- SkinGrinding，简单磨皮（双边滤波，高斯滤波，中值滤波，均值滤波）
- SolderJoint_ConnectedComponents，焊点检测（腐蚀后计算连通域数量）
- EdgeDetection，边缘检测（基于sobel算子和canny算子）
- ImageRectification，简单图像自动矫正（使用投影变换）
- Parts_hough，检测零件图片上的直线（使用两种霍夫变换）
- Wheel_CircularDetection，检测轮毂图片上的一些圆形（轮廓检测，计算圆形度）
- ChipDetection，检测芯片正中央的正方形（中值滤波后连通域标记，筛选正方形）
- CupDetection，检测红色杯盖（画hsv直方图找范围，连通域标记）
- HOG_manual，手动实现HOG算法（图片分块，作梯度直方图，计算欧氏距离）
- Fourier_Transform，图片的傅里叶变换，并能在频谱图上选择一个区域将频率置0，再输出新图片（傅里叶变换，回调函数响应鼠标点击，傅里叶反变换）
- Woven_Remove，消除布纹（傅里叶变换，去除高频区域的连通域，再反变换）
- Hist_Transform，直方图线性变换、单通道gamma矫正、三通道直方图均衡、三通道gamma矫正（LUT查找表）
- Background_Diff，背景差分（内含验证摄像头灰度值高斯分布，手动实现的背景差分（还做了简单的连通域追踪），opencv自动实现的背景差分）

