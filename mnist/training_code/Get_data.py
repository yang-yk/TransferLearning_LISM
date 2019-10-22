# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:21:47 2018

@author: yyk
"""
import scipy.io as sio
import numpy as np
import Option
import tensorflow as tf


def load_test_data(): 
    
    test_data=sio.loadmat(Option.dataset_path+'test_batch_output.mat')
    test_label=sio.loadmat(Option.dataset_path+'test_batch_input.mat')    

    test_data=np.array(test_data['test_batch_output'])
    test_label=np.array(test_label['test_batch_input'])    
    
    #test_number=test_data.shape[0] 
    
    test_label=test_label.astype('float32')   
    
    
    test_data=np.transpose(test_data,[2,1,0])
    test_label=np.transpose(test_label,[2,1,0])    
    
    
    test_data=test_data/np.max(test_data)
    test_label=test_label/np.max(test_label)
    
    test_data=test_data.reshape(list(test_data.shape+(1,))) 
    test_label=test_label.reshape(list(test_label.shape+(1,)))
    
    valid_batch_x=test_data[0:Option.batch_size]
    valid_batch_y=test_label[0:Option.batch_size]
    
    valid_batch_x=valid_batch_x.astype('float32')
    
    return valid_batch_x,valid_batch_y


#读取tfrecord文件，并生成批次  
def read_and_decode(tfrecords_file, batch_size,data_name,label_name):  
    '''''read and decode tfrecord file, generate (image, label) batches 
    Args: 
        tfrecords_file: the directory of tfrecord file 
        batch_size: number of images in each batch 
    Returns: 
        image: 4D tensor - [batch_size, width, height, channel] 
        label: 1D tensor - [batch_size] 
    '''  
    # make an input queue from the tfrecord file  
    #将文件生成一个队列  
    filename_queue = tf.train.string_input_producer([tfrecords_file])  
    #创建一个reader来读取TFRecord文件  
    reader = tf.TFRecordReader()  
    #从文件中独处一个样例。也可以使用read_up_to函数一次性读取多个样例  
    _, serialized_example = reader.read(filename_queue)  
    #解析每一个元素。如果需要解析多个样例，可以用parse_example函数  
    img_features = tf.parse_single_example(  
        serialized_example,  
        features={  
            data_name: tf.FixedLenFeature(shape=[32*32],dtype=tf.float32),  
            label_name : tf.FixedLenFeature(shape=[32*32],dtype=tf.float32),  
        })  
    #tf.decode_raw可以将字符串解析成图像对应的像素数组  
    #train_data = tf.decode_raw(img_features['train_data'], tf.float32)  
    #train_label = tf.decode_raw(img_features['train_label'], tf.float32)  

    #print(train_data)
    print(img_features[data_name])
    train_data=tf.reshape(img_features[data_name],[32,32,1])
    train_label=tf.reshape(img_features[label_name],[32,32,1])
  
    ##########################################################  
    # you can put data augmentation here, I didn't use it  
    ##########################################################  
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.  
  
    
#==============================================================================
#     image_batch, label_batch = tf.train.shuffle_batch([train_data, train_label],  
#                                               batch_size=batch_size,  
#                                               num_threads=4,  
#                                               capacity=30*batch_size,min_after_dequeue=10*batch_size)  
#==============================================================================
      
    image_batch, label_batch = tf.train.batch([train_data, train_label],  
                                              batch_size=batch_size,  
                                              num_threads=1,  
                                              capacity=30*batch_size)  
    
    
    
    return image_batch, label_batch



