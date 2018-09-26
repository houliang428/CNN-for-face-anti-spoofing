import tensorflow as tf
import numpy as np
import os


#%%


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    real = []
    label_real = []
    fake = []
    label_fake = []
#定义一个列表，其中分别存放真照片和假照片的路径和标签值

    for file in os.listdir(file_dir):
    #对路径file_dir中的每个文件逐一进行如下操作
        name = file.split(sep='.')        #把以.隔开的文件字符名称赋值给name
        if name[0]=='real':
            real.append(file_dir + file)  #路径加文件名赋值给real这个列表
            label_real.append(1)          #若为真照片，令其标签为1
        if name[0]=='fake':
            fake.append(file_dir + file)
            label_fake.append(0)
    print('There are %d real images\nThere are %d fake images' %(len(real), len(fake)))
    
    image_list = np.hstack((real, fake))
    #将两个列表real和fake合起来成为一个文件总列表
    label_list = np.hstack((label_real, label_fake))
    
    temp = np.array([image_list, label_list])
    #建立一个数组，第一行是image_list,第二行是label_list,
    #label_list中的元素在生成数组后由int类型变成numpy.str_类型
    temp = temp.transpose()  
    np.random.shuffle(temp)  #将行之间顺序打乱
    
    all_image_list = temp[:, 0]   #所有的图片路径名，temp数组第一列
    all_label_list = temp[:, 1]   #所有标签值，temp数组第二列
    all_label_list = [int(float(i)) for i in all_label_list]
#*****************************************************
#***all_image_list和image_list相比只是问了打乱路径的顺序***
#*****************************************************
    
    return all_image_list,all_label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string) #将image列表转换成tf.string
    label = tf.cast(label, tf.int32)  #将标签列表中的元素int64转换成int32

    input_queue = tf.train.slice_input_producer([image, label])
    #从两个tensorlist image和label中取一个切片，生成输入队列
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3) 
    #图片解码
    image = tf.image.resize_images(image,(image_W,image_H),0)    
    #图像缩放
    image = tf.image.per_image_standardization(image)
    #图像标准化，降低相关性
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    #按顺序从队列中读取数据，队列中的数据始终是一个有序的队列，capacity为队列的长度
    
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


   
