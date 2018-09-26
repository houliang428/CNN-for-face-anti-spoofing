# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:57:57 2018

@author: Liang Hou
"""

import numpy as np
import tensorflow as tf
import model_c
from PIL import Image
import matplotlib.pyplot as plt

def get_one_image():
    '''Randomly pick one image from training data
    Return: ndarray
    '''

    img_dir = 'E:\\houliang\\img5.jpg'

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([96, 96])
    image = np.array(image)
    return image

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    image_array = get_one_image()
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 96, 96, 3])
        
        logit = model_c.inference(image, BATCH_SIZE, N_CLASSES,1.0)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[96, 96, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = 'D:\\train_myself\\model_testc\\'
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==1:
                print('This is a real image with possibility %.6f' %prediction[:, 1])
            else:
                print('This is a fake image with possibility %.6f' %prediction[:, 0])