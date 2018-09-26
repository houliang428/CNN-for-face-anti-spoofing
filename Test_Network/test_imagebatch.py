#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:04:27 2018

@author: houliang
"""

import numpy as np
import tensorflow as tf
import input_getbatch
import model_testc


N_CLASSES = 2
IMG_W = 96
IMG_H = 96
CAPACITY = 2000
MAX_STEP = 795
BATCH_SIZE = 8
                
def run_testing():
    
    val_dir = 'E:\\test_images\\'
    logs_train_dir = 'D:\\train_myself\\model_testc\\' 


    print('Test Images:')
    val,val_label = input_getbatch.get_files(val_dir)
    
    val_batch, val_label_batch = input_getbatch.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
    

    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 96, 96, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = model_testc.inference(x, BATCH_SIZE, N_CLASSES,1.0)
    loss = model_testc.losses(logits, y_) 
    acc = model_testc.evaluation(logits, y_)
    total_loss = 0
    total_acc = 0
    
    saver = tf.train.Saver()      
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x:val_images, y_:val_labels})
                total_loss = total_loss+val_loss
                total_acc = total_acc+val_acc
                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                    
                if step + 1 ==MAX_STEP:
                    print('Total accuracy = %.5f%%'%((total_acc/(step+1)*100.0)))
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)