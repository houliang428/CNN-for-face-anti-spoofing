#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:00:38 2018

@author: houliang
"""

import cv2
import app_console

Video_dir = '' 
store_dir = ''
#视频保存目录以及截取后图像保存地址

real_count = 0
fake_count = 0

for i in range(1, 13): #每个人12段视频，依次读入每段视频
    vc = cv2.VideoCapture(Video_dir+str(i)+'.avi') 
    flag, frame = vc.read()  #flag 是否存在下一帧； frame: 当前视频帧
#计算视频总帧数
    frame_count = 0
    while(flag):
        ret, frame = vc.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    print(frame_count)
    vc.release()

    gap = frame_count//20   #均匀间隔截取20帧
    c = 1   #当前帧序号计数
    
    vc = cv2.VideoCapture(Video_dir+str(i)+'.avi')
    flag, frame = vc.read()
    while (flag):   
            flag, frame = vc.read()
            if (flag == 0):
                break
            if (c % gap == 0):
                if(i<=3):
                    if(i<=2):
                        target_image = app_console.face_crop(frame,240,240)
                    else:
                        target_image = app_console.face_crop(frame,600,800)
                        #高清视频文件人像较大，故截取600*800大小的帧
                    cv2.imwrite(store_dir+'real'+'.' + str(real_count) + '.png',target_image)    #存储为图像
                    real_count = real_count + 1
                else:
                    if(i<=9):
                        target_image = app_console.face_crop(frame,240,240)
                    else:
                        target_image = app_console.face_crop(frame,600,800)
                    cv2.imwrite(store_dir+'fake'+'.' + str(fake_count) + '.png',target_image)    #存储为图像
                    fake_count = fake_count + 1
            c = c + 1
    cv2.waitKey(1)

