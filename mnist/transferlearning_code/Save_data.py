# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:32:20 2018

@author: yyk
"""

import os
import matplotlib.pyplot as plt
import Option
import numpy as np


def define_data():
    train_loss=[]
    train_mse_loss=[]
    train_sharpness_loss=[]
    train_l2_loss=[] 
    
    valid_loss=[]
    valid_mse_loss=[]
    valid_sharpness_loss=[]
    valid_l2_loss=[]
    
    test_loss=[]
    test_mse_loss=[]
    test_sharpness_loss=[]
    test_l2_loss=[]
   
    return train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,test_loss,test_mse_loss,test_sharpness_loss,test_l2_loss

def train_list_append(train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,train_total_loss_,train_l2_loss_,train_mse_loss_,train_sharp_loss_):
     train_loss.append(train_total_loss_)
     train_l2_loss.append(train_l2_loss_)
     train_mse_loss.append(train_mse_loss_)
     train_sharpness_loss.append(train_sharp_loss_)
     return train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss

def valid_list_append(valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,valid_total_loss_,valid_l2_loss_,valid_mse_loss_,valid_sharp_loss_):
     valid_loss.append(valid_total_loss_)
     valid_l2_loss.append(valid_l2_loss_)
     valid_mse_loss.append(valid_mse_loss_)
     valid_sharpness_loss.append(valid_sharp_loss_)
     return valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss

def test_list_append(test_loss,test_mse_loss,test_sharpness_loss,test_l2_loss,test_total_loss_,test_l2_loss_,test_mse_loss_,test_sharp_loss_):
     test_loss.append(test_total_loss_)
     test_l2_loss.append(test_l2_loss_)
     test_mse_loss.append(test_mse_loss_)
     test_sharpness_loss.append(test_sharp_loss_)
     return test_loss,test_mse_loss,test_sharpness_loss,test_l2_loss
     
     
def save_data(train_loss,train_mse_loss,train_sharpness_loss,train_l2_loss,valid_loss,valid_mse_loss,valid_sharpness_loss,valid_l2_loss,\
                test_loss,test_sharpness_loss,test_l2_loss,test_mse_loss):
    print('Saving training data...')
    os.mkdir(Option.result_path+'/data')
    data_path=Option.result_path+'/data'
   
    np.save(data_path+'/train_loss.npy',train_loss)
    np.save(data_path+'/train_sharpness_loss.npy',train_sharpness_loss)
    np.save(data_path+'/train_l2_loss.npy',train_l2_loss)
    np.save(data_path+'/train_mse_loss.npy',train_mse_loss)
   
    print('Saving valid data...')
    np.save(data_path+'/valid_loss.npy',valid_loss)
    np.save(data_path+'/valid_sharpness_loss.npy',valid_sharpness_loss)
    np.save(data_path+'/valid_l2_loss.npy',valid_l2_loss)
    np.save(data_path+'/valid_mse_loss.npy',valid_mse_loss)
    
    
    print('Saving test data...')
    np.save(data_path+'/test_loss.npy',test_loss)
    np.save(data_path+'/test_sharpness_loss.npy',test_sharpness_loss)
    np.save(data_path+'/test_l2_loss.npy',test_l2_loss)
    np.save(data_path+'/test_mse_loss.npy',test_mse_loss)
 
   
#    np.save(data_path+'/label_imgs.npy',valid_batch_y)
#    np.save(data_path+'/predict_imgs.npy',predict_imgs_)   

    


def save_loss_fig(train_loss,valid_loss):
    os.mkdir(Option.result_path+'/fig')
    fig_path=Option.result_path+'/fig'

    plt.plot(train_loss)
    plt.title("train_loss")
    plt.savefig(fig_path+'/train_loss.jpg')
    plt.close()


    plt.plot(valid_loss)
    plt.title('valid_loss')
    plt.savefig(fig_path+'/valid_loss.jpg')
    plt.close()

def save_img(imgs,imagetype,number):
    if imagetype=='test_label':
        if not os.path.exists(Option.result_path+'/label_images'):
            os.mkdir(Option.result_path+'/label_images')
        for i in range(number):
            plt.figure(figsize=(4.32,2.88))
            if i%50==0:
                print('saving labelimage:',i)
            plt.imshow(imgs[i,:,:],cmap='gray',interpolation="bicubic")
            plt.savefig(Option.result_path+'/label_images/'+str(i)+'.jpg')
            plt.close()
               
    
    if imagetype=='test_predict':
        if not os.path.exists(Option.result_path+'/test_images'):
            os.mkdir(Option.result_path+'/test_images')
        for i in range(number):
            plt.figure(figsize=(4.32,2.88))
            if i%50==0:
                print('saving image:',i)
            plt.imshow(imgs[i,:,:],cmap='gray',interpolation="bicubic")
            plt.savefig(Option.result_path+'/test_images/'+str(i)+'.jpg')
            plt.close()
            
            
    if imagetype=='valid_label':
        if not os.path.exists(Option.result_path+'/valid_label_images'):
            os.mkdir(Option.result_path+'/valid_label_images')
        for i in range(number):
            plt.figure(figsize=(4.32,2.88))
            if i%50==0:
                print('saving validlabelimage:',i)
            plt.imshow(imgs[i,:,:],cmap='gray',interpolation="bicubic")
            plt.savefig(Option.result_path+'/valid_label_images/'+str(i)+'.jpg')
            plt.close()
               
    
    if imagetype=='valid_predict':
        if not os.path.exists(Option.result_path+'/valid_test_images'):
            os.mkdir(Option.result_path+'/valid_test_images')
        for i in range(number):
            plt.figure(figsize=(4.32,2.88))
            if i%50==0:
                print('saving validtestimages:',i)
            plt.imshow(imgs[i,:,:],cmap='gray',interpolation="bicubic")
            plt.savefig(Option.result_path+'/valid_test_images/'+str(i)+'.jpg')
            plt.close()

