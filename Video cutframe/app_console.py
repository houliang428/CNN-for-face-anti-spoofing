#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Cropman Console
    ~~~~~~~~~~~~~~~

    Face-aware image cropper application (console interface).

    :copyright: (c) 2014 by Ming YANG.
    :license: WTFPL (Do What the Fuck You Want to Public License).

    Usage:
      app-console.py <input-image> <target-width> <target-height> <target-image>

    Options:
      -h --help     Show this screen.
      --version     Show version.
"""

from cropper import Cropper
import cv2

def face_crop(input_image,target_width,target_height):
    
    cropper = Cropper()
    
    target_image = cropper.crop(input_image, target_width, target_height)
    if target_image is None:
        print ('Cropping failed.')

    return target_image