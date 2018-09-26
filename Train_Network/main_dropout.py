9#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:49:41 2018

@author: houliang
"""


import os
import numpy as np
import tensorflow as tf
import input_getbatch
import model_c
#model_a,model_b and model_c


N_CLASSES = 2
IMG_W = 96  
IMG_H = 96

BATCH_SIZE = 16
CAPACITY = 4000
MAX_STEP = 4000 
learning_rate = 0.0001 


def run_training():
    

    train_dir = 'D:\\database\\train\\'   
    logs_train_dir = 'D:\\train_myself\\model_testc\\'  
    
    train, train_label = input_getbatch.get_files(train_dir)    
    train_batch, train_label_batch = input_getbatch.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
    
    
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])  
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])     
    
    logits = model_c.inference(x, BATCH_SIZE, N_CLASSES,0.5)
    loss = model_c.losses(logits, y_)  
    acc = model_c.evaluation(logits, y_)
    train_op = model_c.trainning(loss, learning_rate)
    
    
             
    with tf.Session() as sess:
        saver = tf.train.Saver()    
        sess.run(tf.global_variables_initializer())    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x:tra_images, y_:tra_labels})
                                       
                if step % 100 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
        
