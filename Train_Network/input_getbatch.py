import tensorflow as tf
import numpy as np
import os


#%%


def get_files(file_dir):

    real = []
    label_real = []
    fake = []
    label_fake = []


    for file in os.listdir(file_dir):

        name = file.split(sep='.')        
        #Divide images named like "fake.1.png "into real or fake groups 
        
        if name[0]=='real':
            real.append(file_dir + file)  
            label_real.append(1)          
        if name[0]=='fake':
            fake.append(file_dir + file)
            label_fake.append(0)
    print('There are %d real images\nThere are %d fake images' %(len(real), len(fake)))
    
    image_list = np.hstack((real, fake))

    label_list = np.hstack((label_real, label_fake))
    
    temp = np.array([image_list, label_list])

    temp = temp.transpose()  
    np.random.shuffle(temp)  
    
    all_image_list = temp[:, 0]  
    all_label_list = temp[:, 1]   
    all_label_list = [int(float(i)) for i in all_label_list]
*****************************************************
  
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
    
    image = tf.cast(image, tf.string) 
    label = tf.cast(label, tf.int32)  

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3) 
    image = tf.image.resize_images(image,(image_W,image_H),0)    
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


   
