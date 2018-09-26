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


real_count = 0
fake_count = 0

for i in range(1, 13): #Each person has 12 videos
    vc = cv2.VideoCapture(Video_dir+str(i)+'.avi') 
    flag, frame = vc.read() 
#Calculating total frames
    frame_count = 0
    while(flag):
        ret, frame = vc.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    print(frame_count)
    vc.release()

    gap = frame_count//20   
    c = 1   
    
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

                    cv2.imwrite(store_dir+'real'+'.' + str(real_count) + '.png',target_image)    
                    real_count = real_count + 1
                else:
                    if(i<=9):
                        target_image = app_console.face_crop(frame,240,240)
                    else:
                        target_image = app_console.face_crop(frame,600,800)
                    cv2.imwrite(store_dir+'fake'+'.' + str(fake_count) + '.png',target_image)   
                    fake_count = fake_count + 1
            c = c + 1
    cv2.waitKey(1)

