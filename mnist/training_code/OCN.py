# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:53:59 2018

@author: yyk
"""
from __future__ import division
import tensorflow as tf
import utils.layers as layers
import Option
import numpy as np
import tensorflow.contrib.slim as slim


def build(inputs,channel,is_training,fine_tuning=True,keep_prob=0.3):
    
    inputs=tf.cast(inputs,dtype=tf.float32,name='inputs')
    print(inputs.shape)
    kernel_shape=[3,3]     

    
    with tf.variable_scope('LSIU_OCN',reuse=tf.AUTO_REUSE):
        
        
        #inputs = tf.layers.batch_normalization(inputs, center = True, scale = True, training = is_training) 
        inputs=tf.layers.batch_normalization(inputs=inputs,center=True,scale=True, training=is_training, fused=False)
       
        is_batch=True
         
        conv1_1=layers.conv_btn(inputs,kernel_shape,Option.feature_num[0],'conv1_1',is_training=is_training,is_batch=is_batch)
        pool1=layers.maxpool(conv1_1,[2,2],'pool1')
   
        conv2_1=layers.conv_btn(pool1,kernel_shape,Option.feature_num[1],'conv2_1',is_training=is_training,is_batch=is_batch)
        pool2=layers.maxpool(conv2_1,[2,2],'pool2') 

        conv3_1=layers.conv_btn(pool2,kernel_shape,Option.feature_num[2],'conv3_1',is_training=is_training,is_batch=is_batch)
        pool3=layers.maxpool(conv3_1,[2,2],'pool3')
    
        conv4_1=layers.conv_btn(pool3,kernel_shape,Option.feature_num[3],'conv4_1',is_training=is_training,is_batch=is_batch)
        pool4=layers.maxpool(conv4_1,[2,2],'pool4')
   
        drop4=layers.dropout(pool4,keep_prob,'drop4')

        upsample5=layers.deconv_upsample(drop4,2,'upsample5')
   
        concat5=layers.concat(upsample5,conv4_1,'concat5')
        conv5_1=layers.conv_btn(concat5,kernel_shape,Option.feature_num[4],'conv5_1',is_training=is_training,is_batch=is_batch)
   
        upsample6=layers.deconv_upsample(conv5_1,2,'upsample6')
   
        concat6=layers.concat(upsample6,conv3_1,'concat6')
        conv6_1=layers.conv_btn(concat6,kernel_shape,Option.feature_num[5],'conv6_1',is_training=is_training,is_batch=is_batch)
    
        upsample7=layers.deconv_upsample(conv6_1,2,'upsample7')
    
        concat7=layers.concat(upsample7,conv2_1,'concat7')
        conv7_1=layers.conv_btn(concat7,kernel_shape,Option.feature_num[6],'conv7_1',is_training=is_training,is_batch=is_batch)
    
        upsample8=layers.deconv_upsample(conv7_1,2,'upsample8')

        concat8=layers.concat(upsample8,conv1_1,'concat8')
        conv8_1=layers.conv_btn(concat8,[3,3],Option.feature_num[7],'conv8_1',is_training=is_training,is_batch=is_batch)
        
        
        
             
        with tf.variable_scope('fine_tuning',reuse=tf.AUTO_REUSE):
             conv9_1=layers.conv_btn(conv8_1,[3,3],64,'conv9_1',is_training=fine_tuning,is_batch=True)    
             #conv9_1=layers.conv_btn(inputs,[3,3],64,'conv9_1',is_training=fine_tuning,is_batch=True)                        
             fc_data=tf.reshape(conv9_1,[Option.batch_size,-1],name='fc_data')
             fc_output = slim.fully_connected(fc_data, 32*32, activation_fn=tf.nn.sigmoid,trainable=True, 
                        scope='fc')
                        
        predict=tf.reshape(fc_output,[-1,32,32,1],name='predict')     

             
             #predict=tf.nn.softmax(predict,name='final_images')
             
             
    tf.summary.image('predicted_images',predict,10)

    return predict
def mse_loss(predict,label,name):   
   
    mse_loss=tf.losses.mean_squared_error(label[:,:,:,0],predict[:,:,:,0])*32*32
    mse_loss=tf.add(0.0,mse_loss,name='mse_loss_op')
    tf.summary.scalar("loss/MSE_LOSS",mse_loss)
    
    return mse_loss
    
def l2_loss(name):    
   
      weights=[var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
     
      '''
      #l2_loss=tf.add_n(tf.nn.l2_loss(w) for w in weights)
      #tf.summary.scalar("loss/weights",l2_loss)    
      '''
     
      for w in weights:
          tf.add_to_collection(tf.GraphKeys.WEIGHTS,w)
      if Option.L1_regularzation==True:
          regularizer=tf.contrib.layers.l1_regularizer(scale=1/Option.data_size)
      else:
          regularizer=tf.contrib.layers.l2_regularizer(scale=30/Option.data_size)
        
      l2_loss=tf.contrib.layers.apply_regularization(regularizer)
      l2_loss=tf.add(0.0,l2_loss,name=name)
      tf.summary.scalar("loss/l2_loss",l2_loss) 
    
      return l2_loss

def sharpness(pre_imgs,lab_imgs,name):    
    
    
    def grad_loss(filters,pre_imgs,lab_imgs):
        tffilter=np.zeros([3,3,1,1])
        tffilter[:,:,0,0]=filters
        pre_grad=tf.nn.conv2d(pre_imgs, tffilter, strides = [1, 1, 1, 1], padding = 'VALID')
        grad_loss=tf.reduce_sum(tf.square(pre_grad))/Option.batch_size
        #lab_grad=tf.nn.conv2d(lab_imgs, tffilter, strides = [1, 1, 1, 1], padding = 'VALID')
        #grad_loss=tf.losses.mean_squared_error(pre_grad,lab_grad)*30*30
        return grad_loss

    Sobel=np.array([[-1,0,1],[-2,0,1],[-1,0,1]])
    SobelT=np.transpose(Sobel)    
    Laplacian=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    
    Sobel_loss=grad_loss(Sobel,pre_imgs,lab_imgs)
    SobelT_loss=grad_loss(SobelT,pre_imgs,lab_imgs)    
    La_loss=grad_loss(Laplacian,pre_imgs,lab_imgs)     
    
    
    sharp_loss=Option.sharpness_weight*(Sobel_loss+ SobelT_loss+La_loss)    
    
    sharp_loss=tf.add(0.0,sharp_loss,name=name)
    tf.summary.scalar("loss/sharp_loss",sharp_loss)
    
    return sharp_loss

def train(loss,learning_rate,learning_rate_decay_steps,learning_rate_decay_rate,global_steps):  

    decayed_learning_rate=tf.train.exponential_decay(learning_rate,global_steps,learning_rate_decay_steps,learning_rate_decay_rate,staircase=True)
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer=tf.train.AdamOptimizer(decayed_learning_rate)
        #optimizer=tf.train.MomentumOptimizer(decayed_learning_rate,momentum=0.9)
        train_op=optimizer.minimize(loss,global_step=global_steps,name='train_op')

    tf.summary.scalar("learning_rate",decayed_learning_rate)
    return train_op

def fine_tune9(loss,learning_rate,learning_rate_decay_steps,learning_rate_decay_rate,global_steps,name):

    varlist=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='LSIU_OCN/fine_tuning')
    decayed_learning_rate=tf.train.exponential_decay(learning_rate,global_steps,learning_rate_decay_steps,learning_rate_decay_rate,staircase=True)

    with tf.control_dependencies(varlist):
        optimizer=tf.train.AdamOptimizer(decayed_learning_rate)
        finetuning_op=optimizer.minimize(loss,global_step=global_steps,var_list=varlist,name=name)

    tf.summary.scalar("fine/learning_rate",decayed_learning_rate)
    return finetuning_op


    
    
    
    
