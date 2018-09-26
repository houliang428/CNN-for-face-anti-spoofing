#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:57:00 2018

@author: houliang
"""

import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
    
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  
    var = _variable_on_cpu(name, shape,
                           tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))

    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    
    # Stores value in the collection with the given name
    tf.add_to_collection('losses', weight_decay)
    return var
#%%
def inference(images, batch_size, n_classes,dropout):
   

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [7,7,3,64],
                                  dtype = tf.float32, 
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases', 
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)

    
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        pool1_dropout=tf.nn.dropout(pool1,dropout,name="pool1_dropout") 
    

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5,5,64,128],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1_dropout, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    

    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling2')
        pool2_dropout=tf.nn.dropout(pool2,dropout,name="pool2_dropout") 
        

    with tf.variable_scope('conv3_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2_dropout, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name='conv3_1')
        
    with tf.variable_scope('conv3_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name='conv3_2')
       
    
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv3_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling3')
        pool3_dropout=tf.nn.dropout(pool3,dropout,name="pool3_dropout") 
            

    with tf.variable_scope('conv4_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,256,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool3_dropout, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name='conv4_1')
        
    with tf.variable_scope('conv4_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name='conv4_2')    
    
    
    with tf.variable_scope('pooling4_lrn') as scope:
        pool4 = tf.nn.max_pool(conv4_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling4')
        pool4_dropout=tf.nn.dropout(pool4,dropout,name="pool4_dropout") 
         
            
    with tf.variable_scope('conv5_1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool4_dropout, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name='conv5_1')
        
    with tf.variable_scope('conv5_2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,512,512],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer(
                                                   factor=1.0,
                                                   mode='FAN_IN',
                                                   uniform=True,
                                                   dtype=tf.float32
                                                           ))
        biases = tf.get_variable('biases',
                                 shape=[512], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv5_1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name='conv5_2')
       

    with tf.variable_scope('pooling5_lrn') as scope:
        pool5 = tf.nn.max_pool(conv5_2, ksize=[1,2,2,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling5')
        pool5_dropout=tf.nn.dropout(pool5,dropout,name="pool5_dropout") 
       
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool5_dropout, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim,128],
                                             stddev=0.1, wd=0.005)
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        local3_dropout=tf.nn.dropout(local3,dropout,name="local3_dropout") 
    
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128,128],
                                             stddev=0.1, wd=0.005)
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3_dropout, weights) + biases, name='local4')
        local4_dropout=tf.nn.dropout(local4,dropout,name="local4_dropout") 
   
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('softmax_linear', shape=[128,n_classes],
                                             stddev=0.1, wd=0.005)
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4_dropout, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%
def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):

  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy
