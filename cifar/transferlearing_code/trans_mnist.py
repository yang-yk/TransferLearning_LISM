# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:06:56 2018

@author: yyk
"""


from __future__ import division

import tensorflow as tf
import Get_data
import OCN
import Option
import os
import numpy as np
import Log
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='1'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

dataset_name='mnist'
result_path='../transderlearning_experiment/'
if not os.path.isdir(result_path):
   os.mkdir(result_path)
   os.mkdir(result_path+dataset_name)
result_path=result_path+dataset_name+'/'

log_path=result_path+'log.txt'

def save_img(imgs,path):
    if not os.path.exists(result_path+path):
       os.mkdir(result_path+path)
    for i in range(imgs.shape[0]): 
        plt.figure(figsize=(4.32,2.88))
        if i%50==0:
           print('saving '+path+':',i)
        plt.imshow((imgs[i,:,:,0]),cmap='gray',interpolation="bicubic")
        plt.savefig(result_path+path+'/'+str(i)+'.jpg')
        plt.close()    


Log.Log(log_path, 'w+', 1) # set log file

fine_tfrecords_trainpath='../transferlearning_dataset/mnist_train_data.tfrecords'
fine_tfrecords_testpath='../transferlearning_dataset/mnist_test_data.tfrecords'


train_data_batch,train_label_batch=Get_data.read_and_decode(fine_tfrecords_trainpath, batch_size=Option.batch_size,data_name='train_data',label_name='train_label')
test_data_batch,test_label_batch=Get_data.read_and_decode(fine_tfrecords_testpath, batch_size=Option.batch_size,data_name='test_data',label_name='test_label')



train_imgs=OCN.build(train_data_batch,Option.channel,is_training=False,fine_tuning=False,keep_prob=0.3)
test_imgs=OCN.build(test_data_batch,Option.channel,is_training=False,fine_tuning=False,keep_prob=1.0)




l2_loss_op=OCN.l2_loss(name='train_l2_loss_op')



train_sharp_loss_op=OCN.sharpness(train_imgs,train_label_batch,name='train_sharp_loss_op')
train_mse_loss_op=OCN.mse_loss(train_imgs,train_label_batch,name='test_mse_loss_op')
train_loss_op=tf.add(l2_loss_op+train_sharp_loss_op+train_mse_loss_op,0.0,name='train_loss')



test_sharp_loss_op=OCN.sharpness(test_imgs,test_label_batch,name='train_sharp_loss_op')
test_mse_loss_op=OCN.mse_loss(test_imgs,test_label_batch,name='test_mse_loss_op')
test_loss_op=tf.add(l2_loss_op+test_sharp_loss_op+test_mse_loss_op,0.0,name='train_loss')


global_step = tf.Variable(0, name = 'global_step', trainable = False)
 
fine_tuing_op=OCN.fine_tune9(train_loss_op,Option.learning_rate,Option.learning_rate_decay_steps,Option.learning_rate_decay_rate,\
  global_step,name='fine_tuing_op')



    
#total_loss_op=tf.add(l2_loss_op+sharp_loss_op+mse_loss_op,0.0,name='train_total_loss_op')  
    
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess=tf.Session(config=config)
sess.run(init_op)


model_path='../training_experiment/2019-10-22-12:34:24/model/'





model_file=tf.train.latest_checkpoint(model_path)
Saver=tf.train.Saver()
Saver.restore(sess,model_file)





coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
step=0

print('<<<<<<<<<<<start>>>>>>>>>>>')
model_imgs=sess.run(test_imgs)
if not os.path.exists(result_path+'data'):
    os.mkdir(result_path+'data')
np.save(result_path+'/data/'+'model_imgs.npy',model_imgs)
save_img(model_imgs,'model_imgs(before finetuning)')


test_imgs_list=[]

try:
    while not coord.should_stop():
        if step%10==0:
            test_imgs_,test_label_imgs_,test_mse_,test_sharp_loss_,test_loss_=sess.run([test_imgs,test_label_batch,\
               test_mse_loss_op,test_sharp_loss_op,test_loss_op])
            print('<<<<<<step: %d test_mse_loss: %3f test_sharp_loss_: %3f test_loss: %3f' %(step,test_mse_,test_sharp_loss_,test_loss_))
            if step==300:
                coord.request_stop()           
        _,train_imgs_,train_label_imgs_,train_mse_,train_sharp_loss_,train_loss_=sess.run([fine_tuing_op,train_imgs,train_label_batch,\
        train_mse_loss_op,train_sharp_loss_op,train_loss_op])
        step=step+1
        test_imgs_list.append(train_imgs_)
        print('step: %d mse_loss: %3f sharp_loss_: %3f train_loss: %3f' %(step,train_mse_,train_sharp_loss_,train_loss_))
        
except  tf.errors.OutOfRangeError:
    print('[INFO    ]\tDone training for %d epochs, %d steps.' % (Option.num_epochs, step))
finally:
         # When done, ask the threads to stop.
    coord.request_stop()
coord.join(threads)
sess.close()



np.save(result_path+'/data/'+'predict_imgs.npy',test_imgs_)
save_img(test_imgs_,'predict_imgs(after finetuning)')

np.save(result_path+'/data/'+'label_imgs.npy',test_label_imgs_)
save_img(test_label_imgs_,'label_imgs')









        
