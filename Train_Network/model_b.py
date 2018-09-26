#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:53:55 2018

@author: houliang
"""

import tensorflow as tf

#%%
def inference(images, batch_size, n_classes,dropout):
    '''建立模型
    参数:
        图像: 图像 batch, 4维 tensor, tf.float32型, [batch_size, 图像宽度, 图像高度, 通道数]
    返回值:
        输出最后网络计算出的logits, 浮点型, [batch_size, n_classes]
    '''    
    #第一层卷积层
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [7,7,3,64],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #第一层池化层   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
    
    #第二层卷积层
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5,5,64,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    #第二层池化层
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling2')
        
    #第三层卷积层
    with tf.variable_scope('conv3_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name='conv3_1')
        
    with tf.variable_scope('conv3_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name='conv3_2')
    
    
    #第三层池化层
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv3_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling3')
        
    
    #第四层卷积层
    with tf.variable_scope('conv4_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool3, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name='conv4_1')
        
    with tf.variable_scope('conv4_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name='conv4_2')
    
    
    #第四层池化层
    with tf.variable_scope('pooling4_lrn') as scope:
        pool4 = tf.nn.max_pool(conv4_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling4')
        
    #第五层卷积层
    with tf.variable_scope('conv5_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool4, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name='conv5_1')
        
    with tf.variable_scope('conv5_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv5_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name='conv5_2')
    
    
    #第五层池化层 
    with tf.variable_scope('pooling5_lrn') as scope:
        pool5 = tf.nn.max_pool(conv5_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling5')
    
    
    #卷积层到全连接层
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        local3_dropout=tf.nn.dropout(local3,dropout,name="local3_dropout") 

    #全连接层
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3_dropout, weights) + biases, name='local4')
        local4_dropout=tf.nn.dropout(local4,dropout,name="local4_dropout") 

     
        
    #全连接层到softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4_dropout, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%
def losses(logits, labels):
    '''损失计算
    参数:
        logits: 网络最后输出的计算结果, 浮点型, [batch_size, n_classes]
        labels: 图像真实标签, 整型, [batch_size]  
    返回值:
        浮点型的损失值
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''模型训练
    输入参数:
        loss: 从losses（）函数中得到的损失值
    返回值:
        train_op: 训练操作
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
    '''计算当前batch的准确值
    参数:
        logits: 网络最后计算出来的值 - [batch_size, NUM_CLASSES].
        labels: 实际标签
    返回值:
        当前batch的准确值
    '''
    with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy